// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! [`PageIndexProvider`]: per-column-chunk access to page indexes.

use crate::file::metadata::ParquetMetaData;
use crate::file::page_index::column_index::ColumnIndexMetaData;
use crate::file::page_index::offset_index::OffsetIndexMetaData;
use std::fmt::Debug;

/// Provides page-index entries for individual column chunks.
///
/// Readers consume page indexes one column chunk at a time (identified by
/// row-group index and leaf-column index). This trait decouples that access
/// from [`ParquetMetaData`]'s dense `[row_group][column]` matrices so that
/// page indexes can come from other sources, for example:
///
/// - A cache of individually decoded entries, populated across queries with
///   only the columns each query needed.
/// - A lazily decoding source that materializes entries on first access.
///
/// [`ParquetMetaData`] implements this trait by reading its embedded page
/// indexes, so existing code keeps working unchanged.
///
/// # Semantics of `None`
///
/// `None` means the entry is *not available* — either the file does not
/// contain it or it was not loaded. Consumers must fall back to reading the
/// affected column chunk without page-index optimizations. The Parquet
/// footer ([`ColumnChunkMetaData::column_index_offset`] /
/// [`ColumnChunkMetaData::offset_index_offset`]) remains the source of truth
/// for whether an index exists in the file.
///
/// [`ColumnChunkMetaData::column_index_offset`]: crate::file::metadata::ColumnChunkMetaData::column_index_offset
/// [`ColumnChunkMetaData::offset_index_offset`]: crate::file::metadata::ColumnChunkMetaData::offset_index_offset
pub trait PageIndexProvider: Debug + Send + Sync {
    /// Returns the column index (per-page statistics) for one column chunk,
    /// if available.
    fn column_index(
        &self,
        row_group_index: usize,
        column_index: usize,
    ) -> Option<&ColumnIndexMetaData>;

    /// Returns the offset index (page locations) for one column chunk,
    /// if available.
    fn offset_index(
        &self,
        row_group_index: usize,
        column_index: usize,
    ) -> Option<&OffsetIndexMetaData>;

    /// Returns `true` if an offset index is available for every column of
    /// `row_group_index` with `num_columns` columns.
    ///
    /// Some consumers plan an entire row group at once and need all of its
    /// offset indexes; the default implementation probes each column.
    fn has_offset_indexes(&self, row_group_index: usize, num_columns: usize) -> bool {
        (0..num_columns).all(|column| self.offset_index(row_group_index, column).is_some())
    }
}

/// [`ParquetMetaData`] provides page indexes from its embedded dense
/// matrices.
///
/// `ColumnIndexMetaData::NONE` cells (written when a file lacks min/max
/// information) are reported as `None`.
impl PageIndexProvider for ParquetMetaData {
    fn column_index(
        &self,
        row_group_index: usize,
        column_index: usize,
    ) -> Option<&ColumnIndexMetaData> {
        let index = ParquetMetaData::column_index(self)?
            .get(row_group_index)?
            .get(column_index)?;
        match index {
            ColumnIndexMetaData::NONE => None,
            _ => Some(index),
        }
    }

    fn offset_index(
        &self,
        row_group_index: usize,
        column_index: usize,
    ) -> Option<&OffsetIndexMetaData> {
        // Old writer versions produced `Some(vec![])` for row groups without
        // a page index; `get` returns `None` for those as well.
        ParquetMetaData::offset_index(self)?
            .get(row_group_index)?
            .get(column_index)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basic::Type as PhysicalType;
    use crate::file::metadata::{
        ColumnChunkMetaData, FileMetaData, OffsetIndexBuilder, ParquetMetaData, RowGroupMetaData,
    };
    use crate::schema::types::{SchemaDescriptor, Type};
    use std::sync::Arc;

    fn footer(num_row_groups: usize) -> ParquetMetaData {
        let field = Arc::new(
            Type::primitive_type_builder("value", PhysicalType::INT32)
                .build()
                .unwrap(),
        );
        let schema = Arc::new(
            Type::group_type_builder("schema")
                .with_fields(vec![field])
                .build()
                .unwrap(),
        );
        let schema = Arc::new(SchemaDescriptor::new(schema));
        let row_groups = (0..num_row_groups)
            .map(|_| {
                let column = ColumnChunkMetaData::builder(schema.column(0))
                    .set_num_values(10)
                    .build()
                    .unwrap();
                RowGroupMetaData::builder(Arc::clone(&schema))
                    .set_num_rows(10)
                    .set_column_metadata(vec![column])
                    .build()
                    .unwrap()
            })
            .collect();
        let file = FileMetaData::new(1, 10, None, None, schema, None);
        ParquetMetaData::new(file, row_groups)
    }

    fn offset_index_entry() -> OffsetIndexMetaData {
        let mut builder = OffsetIndexBuilder::new();
        builder.append_row_count(10);
        builder.append_offset_and_size(100, 50);
        builder.build()
    }

    #[test]
    fn metadata_without_indexes_provides_nothing() {
        let metadata = footer(1);
        let provider: &dyn PageIndexProvider = &metadata;
        assert!(provider.column_index(0, 0).is_none());
        assert!(provider.offset_index(0, 0).is_none());
        assert!(!provider.has_offset_indexes(0, 1));
    }

    #[test]
    fn metadata_serves_embedded_indexes_and_hides_none_cells() {
        let metadata = footer(2)
            .into_builder()
            .set_column_index(Some(vec![
                vec![ColumnIndexMetaData::NONE],
                vec![ColumnIndexMetaData::NONE],
            ]))
            .set_offset_index(Some(vec![vec![offset_index_entry()], vec![]]))
            .build();
        let provider: &dyn PageIndexProvider = &metadata;

        // NONE cells are "not available"
        assert!(provider.column_index(0, 0).is_none());
        // present offset index is served per cell
        assert_eq!(
            provider.offset_index(0, 0).unwrap().page_locations()[0].offset,
            100
        );
        assert!(provider.has_offset_indexes(0, 1));
        // legacy empty row-group entry behaves as missing
        assert!(provider.offset_index(1, 0).is_none());
        assert!(!provider.has_offset_indexes(1, 1));
        // out of bounds is None, not a panic
        assert!(provider.offset_index(9, 9).is_none());
    }
}

/// Tests that a custom, sparse [`PageIndexProvider`] can drive the Arrow
/// readers while the (shared) [`ParquetMetaData`] carries no page indexes.
#[cfg(all(test, feature = "arrow"))]
mod arrow_tests {
    use super::*;
    use crate::arrow::ArrowWriter;
    use crate::arrow::ProjectionMask;
    use crate::arrow::arrow_reader::{
        ArrowReaderMetadata, ArrowReaderOptions, ParquetRecordBatchReaderBuilder, RowSelection,
        RowSelector,
    };
    use crate::file::metadata::{PageIndexPolicy, ParquetMetaDataReader};
    use crate::file::properties::WriterProperties;
    use arrow_array::{ArrayRef, Int32Array, RecordBatch};
    use arrow_schema::{DataType, Field, Schema};
    use bytes::Bytes;
    use std::collections::HashMap;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Sparse provider over individually stored entries, with a hit counter.
    /// Mimics a cache that decoded only the entries a query needs.
    #[derive(Debug, Default)]
    struct SparseProvider {
        offset_indexes: HashMap<(usize, usize), OffsetIndexMetaData>,
        hits: AtomicUsize,
    }

    impl PageIndexProvider for SparseProvider {
        fn column_index(
            &self,
            _row_group_index: usize,
            _column_index: usize,
        ) -> Option<&ColumnIndexMetaData> {
            None
        }

        fn offset_index(
            &self,
            row_group_index: usize,
            column_index: usize,
        ) -> Option<&OffsetIndexMetaData> {
            let entry = self.offset_indexes.get(&(row_group_index, column_index));
            if entry.is_some() {
                self.hits.fetch_add(1, Ordering::Relaxed);
            }
            entry
        }
    }

    /// Writes columns `a`, `b`, `c` (0..8, 100..108, 200..208) in two row
    /// groups of four rows each.
    fn write_file() -> Bytes {
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int32, false),
            Field::new("b", DataType::Int32, false),
            Field::new("c", DataType::Int32, false),
        ]));
        let props = WriterProperties::builder()
            .set_max_row_group_row_count(Some(4))
            .build();
        let mut buf = Vec::new();
        let mut writer = ArrowWriter::try_new(&mut buf, Arc::clone(&schema), Some(props)).unwrap();
        let column =
            |base: i32| -> ArrayRef { Arc::new(Int32Array::from_iter_values(base..base + 8)) };
        let batch =
            RecordBatch::try_new(schema, vec![column(0), column(100), column(200)]).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();
        Bytes::from(buf)
    }

    #[test]
    fn sparse_provider_supplies_page_locations_without_metadata_indexes() {
        let file = write_file();

        // Footer only: the shared metadata never carries page indexes.
        let footer = Arc::new(
            ParquetMetaDataReader::new()
                .with_page_index_policy(PageIndexPolicy::Skip)
                .parse_and_finish(&file)
                .unwrap(),
        );
        assert!(footer.column_index().is_none());
        assert!(footer.offset_index().is_none());

        // Simulate a cache that decoded only what the scan needs: offset
        // indexes for columns a and c of row group 1. (Stands in for a
        // future scoped decode API; here they are extracted from a separate
        // full parse.)
        let full = ParquetMetaDataReader::new()
            .with_page_index_policy(PageIndexPolicy::Required)
            .parse_and_finish(&file)
            .unwrap();
        let full_offset_index = full.offset_index().unwrap();
        let mut provider = SparseProvider::default();
        for column in [0, 2] {
            provider
                .offset_indexes
                .insert((1, column), full_offset_index[1][column].clone());
        }
        let provider = Arc::new(provider);

        // SELECT a, c FROM file WHERE <rows 1..3 of row group 1>
        let reader_metadata =
            ArrowReaderMetadata::try_new(Arc::clone(&footer), ArrowReaderOptions::new())
                .unwrap()
                .with_page_index_provider(Arc::clone(&provider) as _);
        let projection = ProjectionMask::leaves(footer.file_metadata().schema_descr(), [0, 2]);
        let selection = RowSelection::from(vec![RowSelector::skip(1), RowSelector::select(2)]);
        let reader = ParquetRecordBatchReaderBuilder::new_with_metadata(file, reader_metadata)
            .with_row_groups(vec![1])
            .with_projection(projection)
            .with_row_selection(selection)
            .build()
            .unwrap();

        let batches: Vec<RecordBatch> = reader.collect::<Result<_, _>>().unwrap();
        let rows: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(rows, 2);

        // Values are from row group 1 (a starts at 4, c at 204), rows 1..3.
        let a = batches[0].column(0).as_any().downcast_ref::<Int32Array>();
        assert_eq!(a.unwrap().values(), &[5, 6]);
        let c = batches[0].column(1).as_any().downcast_ref::<Int32Array>();
        assert_eq!(c.unwrap().values(), &[205, 206]);

        // The reader consulted the sparse provider for page locations.
        assert!(provider.hits.load(Ordering::Relaxed) > 0);
    }
}
