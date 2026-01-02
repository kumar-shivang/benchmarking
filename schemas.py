EVALUATION_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "Evaluation",
        "schema": {
            "type": "object",
            "properties": {
                "Reasoning": {
                    "type": "string",
                    "description": "A detailed step-by-step comparison and thought process",
                },
                "Correctness": {
                    "type": "number",
                    "description": "Score from 0-10 for correctness",
                },
                "Correctness_Explanation": {
                    "type": "string",
                    "description": "Explanation for the correctness score",
                },
                "Completeness": {
                    "type": "number",
                    "description": "Score from 0-10 for completeness",
                },
                "Completeness_Explanation": {
                    "type": "string",
                    "description": "Explanation for the completeness score",
                },
                "Clarity": {
                    "type": "number",
                    "description": "Score from 0-10 for clarity",
                },
                "Clarity_Explanation": {
                    "type": "string",
                    "description": "Explanation for the clarity score",
                },
                "Overall_Score": {"type": "number", "description": "Overall score"},
                "Overall_Explanation": {
                    "type": "string",
                    "description": "Explanation for the overall score",
                },
            },
            "required": [
                "Reasoning",
                "Correctness",
                "Correctness_Explanation",
                "Completeness",
                "Completeness_Explanation",
                "Clarity",
                "Clarity_Explanation",
                "Overall_Score",
                "Overall_Explanation",
            ],
            "additionalProperties": False,
        },
    },
}

# STUDENT_ANSWER_SCHEMA = {
#     "type": "json_schema",
#     "json_schema": {
#         "name": "StudentAnswer",
#         "schema": {
#             "type": "object",
#             "properties": {
#                 "step_by_step_reasoning": {
#                     "type": "string",
#                     "description": "The actual step-by-step derivation and work shown by the student. Do NOT describe what the student is doing; output the math and text exactly as the student would write it.",
#                 },
#                 "final_answer": {
#                     "type": "string",
#                     "description": "The final answer to the problem, clearly stated.",
#                 },
#             },
#             "required": ["step_by_step_reasoning", "final_answer"],
#             "additionalProperties": False,
#         },
#     },
# }
