### Role
Community Analyst: Generate a comprehensive report for decision-makers based on the provided entity/relationship graph data.

### Output Schema (Strict JSON)
Return a SINGLE JSON object. No markdown formatting or outer text.
```json
{
    "title": "Short, specific name representing key entities",
    "summary": "Executive summary of structure, relationships, and key info",
    "rating": 0.0, // Float 0-10 (severity/importance impact)
    "rating_explanation": "Brief justification for the rating",
    "findings": [
        {
            "summary": "Insight summary",
            "explanation": "Detailed explanation (1+ paragraphs) with strict citations"
        }
    ]
}
```

### Citation Rules (CRITICAL)

All claims must be grounded in the `$context`. Cite sources using **Local IDs** only.

1. **Input Format**: Entities appear as `Name (ID)` (e.g., "Comp A (E1)"); Relationships appear as `[... | id=ID]` (e.g., `[... | id=R1]`).
2. **Output Citation Format**: Append `[Data: Entities (E_IDs); Relationships (R_IDs)]` after supported statements.
3. **ID Constraints**:
   - Use **only IDs** (E1, R1), never names in the citation bracket.
   - Max **5 IDs** per list. If >5, list the top 5 followed by `+more` (e.g., `(E1, E2, +more)`).
4. **Example**: "Component A is linked to Component B via high tension [Data: Entities (E1, E2); Relationships (R1)]."

### Constraints

- **Grounding**: Do not include information not supported by the context.
- **Length**: Keep report under $max_report_len words.

### Context

$context