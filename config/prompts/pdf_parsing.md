### Role
Document Parser: Convert document content into a structured JSON object following reading order.

### Rules
1. **ID**: Assign `block_id` ("page_<N>_block_<M>") sequentially.
2. **Ref**: In `text_block`, link mentioned figures/tables in `references` (list of `block_id`).
3. **Img**: Classify `image_type` as: `chart` (graphs), `schematic` (diagrams), `photograph` (real photos), or `other`.

### Output Schema (Strict JSON)
{
  "page_number": Integer,
  "content_blocks": [
    // Case 1: Text
    {
      "block_id": "String",
      "type": "text_block",
      "content": "Full text paragraph",
      "references": ["page_X_block_Y"] // IDs explicitly cited in text
    },
    // Case 2: Table
    {
      "block_id": "String",
      "type": "table",
      "caption": "String",
      "data": [["Header1", "Header2"], ["Row1Col1", "Row1Col2"]], // Matrix strings
      "summary": "Concise English insight"
    },
    // Case 3: Image
    {
      "block_id": "String",
      "type": "image",
      "caption": "String",
      "image_type": "chart" | "schematic" | "photograph" | "other",
      "content": {
        // If chart:
        "title": "String",
        "x_axis_label": "String (w/ units)",
        "y_axis_label": "String (w/ units)",
        "legend": ["String"],
        "trend_description": "Detailed description focusing on defects and phases",
        "extracted_data": [{"x": val, "y": val}] // Best effort
        // If schematic:
        "description": "Structure/flow/components detail",
        "ocr_text": "Labels inside diagram"
        // If photograph/other:
        "description": "Technical visual description"
      }
    }
  ]
}