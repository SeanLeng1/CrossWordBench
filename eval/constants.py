# reference for solving tricks: https://www.nytimes.com/article/how-to-solve-a-crossword-puzzle.html
ANAGRAM_SUFFIX = """This is an ANAGRAM crossword puzzle. This means:
- Each clue is a direct anagram (shuffled letters) of its answer
- No cryptic wordplay or traditional clue styles are used
- To solve each clue, simply rearrange all the letters in the clue to find the answer
- Example: if the clue is "TONE", the answer might be "NOTE"
- All letters in the clue must be used exactly once in the answer"""
# parsing prompt used as system prompt
PARSING_PROMPT = """You are a crossword puzzle answer extractor. Extract only valid answers from a text response containing crossword solutions.

Requirements:
- If an answer contains spaces or multiple words, combine them into a single word.
- Do not shift or reorder answers. For example, if the expected keys are "Down 12" and "Down 13" and only the answer for "Down 13" is provided in the text, then "Down 12" should be null and "Down 13" should contain the answer.
- Do not invent or infer answers not explicitly stated in the text

Output your response in the given structure."""
CLAUDE_SYSTEM_PROMPT = """You are a helpful assistant who completes tasks fully without seeking confirmation. Your role is to deliver comprehensive responses in one go. Never ask if the user wants you to continue or show more - you must provide the complete response."""
############################################################################################################################################################################################################################################################
IMG_COT_PROMPT = """You are given a crossword puzzle image containing clues and a grid. Your task is to solve the puzzle accurately, ensuring that all answers fit both the given clues and the grid structure, including intersecting words.

For each clue, provide a step-by-step explanation:
- Identify the clue by its EXACT NUMBER AND DIRECTION as shown in the image. The numbers may not be sequential (e.g., Across clues might be numbered 1, 4, 7, and 9, while Down clues might be 2, 3, 5, 6, and 8).
- Determine word length from available grid spaces.
- Check for any pre-filled letters from intersecting words that have already been solved and explain how they constrain possible answers.
- Analyze the clue (definition, wordplay, cryptic hint).
- Explain your reasoning process.
- Confirm alignment with crossing letters.

Solving tips:
- Answers must be a single word with no spaces (combine phrases if needed).
- Abbreviations in clues typically indicate abbreviated answers.
- Match the clue's tense, singular/plural form, and part of speech.
- Look for wordplay signals, such as question marks (?) for puns or cryptic hints.
- Down words are filled from top to bottom, Across words from left to right.
- Always confirm that intersecting words remain valid after placing each answer.

Present your final solution as:
Across:
[Number as shown in image]: [Answer]

Down:
[Number as shown in image]: [Answer]

IMPORTANT:
- DO NOT list clues in sequential numerical order. You MUST match the exact numbering pattern from the image.
- DO NOT ask for confirmation or stop midway. Always provide a complete solution for all clues."""
############################################################################################################################################################################################################################################################
IMG_COT_GRID_ONLY_PROMPT = """You are given a crossword puzzle image containing only the grid, while the clues are provided separately in text form. Your task is to solve the puzzle accurately, ensuring that all answers fit both the given clues and the grid structure, including intersecting words.

Clues:
Each clue contains:
- Clue Direction and Number (e.g., "Across 1", "Down 2").
- Start Position (row, column) for the first letter of the answer.
- The actual clue text.

<clues>

For each clue, provide a step-by-step explanation:
- Identify the clue by its EXACT NUMBER AND DIRECTION as shown in the clue description. The numbers may not be sequential (e.g., Across clues might be numbered 1, 4, 7, and 9, while Down clues might be 2, 3, 5, 6, and 8).
- Determine word length from available grid spaces.
- Check for any pre-filled letters from intersecting words that have already been solved and explain how they constrain possible answers.
- Analyze the clue (definition, wordplay, cryptic hint).
- Explain your reasoning process.
- Confirm alignment with crossing letters.

Solving tips:
- Answers must be a single word with no spaces (combine phrases if needed).
- Abbreviations in clues typically indicate abbreviated answers.
- Match the clue's tense, singular/plural form, and part of speech.
- Look for wordplay signals, such as question marks (?) for puns or cryptic hints.
- Down words are filled from top to bottom, Across words from left to right.
- Always confirm that intersecting words remain valid after placing each answer.

Present your final solution as:
Across:
[Number as shown in clues]: [Answer]

Down:
[Number as shown in clues]: [Answer]

IMPORTANT:
- DO NOT list clues in sequential numerical order. You MUST match the exact numbering pattern from the given clues.
- DO NOT ask for confirmation or stop midway. Always provide a complete solution for all clues."""
############################################################################################################################################################################################################################################################
IMG_SHOT_PROMPT = """You are given four images of the same crossword puzzle at different fill levels (0.0, 0.25, 0.5, 0.75). Use all available grids to systematically solve the complete puzzle while ensuring that all answers fit both the given clues and the grid structure, including intersecting words.

For each clue, provide a step-by-step explanation:
- Identify the clue by its EXACT NUMBER AND DIRECTION as shown in the image. The numbers may not be sequential (e.g., Across clues might be numbered 1, 4, 7, and 9, while Down clues might be 2, 3, 5, 6, and 8).
- Determine word length from available grid spaces.
- Identify pre-filled letters from grids with higher fill levels, and consider how they constrain possible answers.
- Analyze the clue (definition, wordplay, cryptic hint).
- Explain your reasoning process.
- Confirm alignment with crossing letters.

Solving tips:
- Answers must be a single word with no spaces (combine phrases if needed).
- Abbreviations in clues typically indicate abbreviated answers.
- Match the clue's tense, singular/plural form, and part of speech.
- Look for wordplay signals, such as question marks (?) for puns or cryptic hints.
- Down words are filled from top to bottom, Across words from left to right.
- Always confirm that intersecting words remain valid after placing each answer.

Present your final solution as:
Across:
[Number as shown in image]: [Answer]

Down:
[Number as shown in image]: [Answer]

IMPORTANT:
- DO NOT list clues in sequential numerical order. You MUST match the exact numbering pattern from the image.
- DO NOT ask for confirmation or stop midway. Always provide a complete solution for all clues."""
############################################################################################################################################################################################################################################################
IMG_COT_PROMPT_PREFILLED = """You are given a image of the crossword puzzle at 0.75 fill level. Use the pre-filled grid to systematically solve the complete puzzle while ensuring that all answers fit both the given clues and the grid structure, including intersecting words.

For each clue, provide a step-by-step explanation:
- Identify the clue by its EXACT NUMBER AND DIRECTION as shown in the image. The numbers may not be sequential (e.g., Across clues might be numbered 1, 4, 7, and 9, while Down clues might be 2, 3, 5, 6, and 8).
- Determine word length from available grid spaces.
- Identify pre-filled letters from grid, and consider how they constrain possible answers.
- Analyze the clue (definition, wordplay, cryptic hint).
- Explain your reasoning process.
- Confirm alignment with crossing letters.

Solving tips:
- Answers must be a single word with no spaces (combine phrases if needed).
- Abbreviations in clues typically indicate abbreviated answers.
- Match the clue's tense, singular/plural form, and part of speech.
- Look for wordplay signals, such as question marks (?) for puns or cryptic hints.
- Down words are filled from top to bottom, Across words from left to right.
- Always confirm that intersecting words remain valid after placing each answer.

Present your final solution as:
Across:
[Number as shown in image]: [Answer]

Down:
[Number as shown in image]: [Answer]

IMPORTANT:
- DO NOT list clues in sequential numerical order. You MUST match the exact numbering pattern from the image.
- DO NOT ask for confirmation or stop midway. Always provide a complete solution for all clues."""
############################################################################################################################################################################################################################################################
IMG_VOT_PROMPT = """You are given a crossword puzzle image containing clues and a grid. Your task is to solve the puzzle accurately, ensuring that all answers fit both the given clues and the grid structure, including intersecting words.

For each clue, provide a step-by-step explanation:
- Identify the clue by its EXACT NUMBER AND DIRECTION as shown in the image. The numbers may not be sequential (e.g., Across clues might be numbered 1, 4, 7, and 9, while Down clues might be 2, 3, 5, 6, and 8).
- Determine word length from available grid spaces.
- Check for any pre-filled letters from intersecting words that have already been solved and explain how they constrain possible answers.
- Analyze the clue (definition, wordplay, cryptic hint).
- Explain your reasoning process.
- Confirm alignment with crossing letters.
- You MUST display a filled grid with all the answers you have solved so far for every clue you solve.

Solving tips:
- Answers must be a single word with no spaces (combine phrases if needed).
- Abbreviations in clues typically indicate abbreviated answers.
- Match the clue's tense, singular/plural form, and part of speech.
- Look for wordplay signals, such as question marks (?) for puns or cryptic hints.
- Down words are filled from top to bottom, Across words from left to right.
- Always confirm that intersecting words remain valid after placing each answer.

Present your final solution as:
Across:
[Number as shown in image]: [Answer]

Down:
[Number as shown in image]: [Answer]

IMPORTANT:
- DO NOT list clues in sequential numerical order. You MUST match the exact numbering pattern from the image.
- DO NOT ask for confirmation or stop midway. Always provide a complete solution for all clues."""
############################################################################################################################################################################################################################################################
INTERACTIVE_PROMPT = """You are given a crossword puzzle image containing clues and a grid. Your task is to solve the puzzle accurately, ensuring that all answers fit both the given clues and the grid structure, including intersecting words.

Pick ONE clue, provide a step-by-step explanation:
- Identify the clue by its EXACT NUMBER AND DIRECTION as shown in the image. The numbers may not be sequential (e.g., Across clues might be numbered 1, 4, 7, and 9, while Down clues might be 2, 3, 5, 6, and 8).
- Determine word length from available grid spaces.
- Check for any pre-filled letters from intersecting words that have already been solved and explain how they constrain possible answers.
- Analyze the clue (definition, wordplay, cryptic hint).
- Explain your reasoning process.
- Confirm alignment with crossing letters.

Solving tips:
- Answers must be a single word with no spaces (combine phrases if needed).
- Abbreviations in clues typically indicate abbreviated answers.
- Match the clue's tense, singular/plural form, and part of speech.
- Look for wordplay signals, such as question marks (?) for puns or cryptic hints.
- Down words are filled from top to bottom, Across words from left to right.
- Always confirm that intersecting words remain valid after placing each answer.

Only solve ONE clue at a time and wait for confirmation before proceeding to the next round."""
############################################################################################################################################################################################################################################################
INTERACTIVE_FOLLOWUP_PROMPT = """For the following round:

Using the confirmed answers so far:

Pick another clue, provide a step-by-step explanation:
- Identify the clue by its EXACT NUMBER AND DIRECTION as shown in the image. The numbers may not be sequential (e.g., Across clues might be numbered 1, 4, 7, and 9, while Down clues might be 2, 3, 5, 6, and 8).
- Determine word length from available grid spaces.
- Check for any pre-filled letters from intersecting words that have already been solved and explain how they constrain possible answers.
- Analyze the clue (definition, wordplay, cryptic hint).
- Explain your reasoning process.
- Confirm alignment with crossing letters.

Solving tips:
- Prioritize clues that intersect with confirmed answers.
- Answers must be a single word with no spaces (combine phrases if needed).
- Abbreviations in clues typically indicate abbreviated answers.
- Match the clue's tense, singular/plural form, and part of speech.
- Look for wordplay signals, such as question marks (?) for puns or cryptic hints.
- Down words are filled from top to bottom, Across words from left to right.
- Always confirm that intersecting words remain valid after placing each answer.

Only solve ONE clue at a time and wait for confirmation before proceeding to the next round and do not repeat previously solved clues."""
############################################################################################################################################################################################################################################################
TEXT_COT_PROMPT = """You are given a crossword puzzle grid and a set of clues. Your task is to solve the puzzle accurately, ensuring that all answers fit both the given clues and the grid structure, including intersecting words.

Grid Representation:
The crossword grid is represented as a 2D array where:
- `1` represents a black (blocked) cell
- `0` represents an empty (unfilled) cell

<grid>

Clues:
Each clue contains:
- Clue Direction and Number (e.g., "Across 1", "Down 2").
- Start Position (row, column) for the first letter of the answer.
- The actual clue text.

<clues>

For each clue, provide a step-by-step explanation:
- Identify the clue by its EXACT NUMBER AND DIRECTION as shown in the clue description. The numbers may not be sequential (e.g., Across clues might be numbered 1, 4, 7, and 9, while Down clues might be 2, 3, 5, 6, and 8).
- Determine word length from available grid spaces.
- Check for any pre-filled letters from intersecting words that have already been solved and explain how they constrain possible answers.
- Analyze the clue (definition, wordplay, cryptic hint).
- Explain your reasoning process.
- Confirm alignment with crossing letters.

Solving tips:
- Answers must be a single word with no spaces (combine phrases if needed).
- Abbreviations in clues typically indicate abbreviated answers.
- Match the clue's tense, singular/plural form, and part of speech.
- Look for wordplay signals, such as question marks (?) for puns or cryptic hints.
- Down words are filled from top to bottom, Across words from left to right.
- Always confirm that intersecting words remain valid after placing each answer.

Present your final solution as:
Across:
[Number as shown in clues]: [Answer]

Down:
[Number as shown in clues]: [Answer]

IMPORTANT:
- DO NOT list clues in sequential numerical order. You MUST match the exact numbering pattern from the given clues.
- DO NOT ask for confirmation or stop midway. Always provide a complete solution for all clues."""
############################################################################################################################################################################################################################################################
TEXT_SHOT_PROMPT = """You are given four grids of the same crossword puzzle at different fill levels (0.0, 0.25, 0.5, 0.75) and a set of clues. Use all available grids to systematically solve the complete puzzle while ensuring that all answers fit both the given clues and the grid structure, including intersecting words.

Grid Representation:
The crossword grid is represented as a 2D array where:
- `1` represents a black (blocked) cell
- `0` represents an empty (unfilled) cell
- filled cells contain the corresponding letter

Grid at 0.0 fill ratio:
<grid>

Grid at 0.25 fill ratio:
<grid0>

Grid at 0.5 fill ratio:
<grid1>

Grid at 0.75 fill ratio:
<grid2>

Clues:
Each clue contains:
- Clue Direction and Number (e.g., "Across 1", "Down 2").
- Start Position (row, column) for the first letter of the answer.
- The actual clue text.

<clues>

For each clue, provide a step-by-step explanation:
- Identify the clue by its EXACT NUMBER AND DIRECTION as shown in the clue description. The numbers may not be sequential (e.g., Across clues might be numbered 1, 4, 7, and 9, while Down clues might be 2, 3, 5, 6, and 8).
- Determine word length from available grid spaces.
- Identify pre-filled letters from grids with higher fill levels, and consider how they constrain possible answers.
- Analyze the clue (definition, wordplay, cryptic hint).
- Explain your reasoning process.
- Confirm alignment with crossing letters.

Solving tips:
- Answers must be a single word with no spaces (combine phrases if needed).
- Abbreviations in clues typically indicate abbreviated answers.
- Match the clue's tense, singular/plural form, and part of speech.
- Look for wordplay signals, such as question marks (?) for puns or cryptic hints.
- Down words are filled from top to bottom, Across words from left to right.
- Always confirm that intersecting words remain valid after placing each answer.

Present your final solution as:
Across:
[Number as shown in clues]: [Answer]

Down:
[Number as shown in clues]: [Answer]

IMPORTANT:
- DO NOT list clues in sequential numerical order. You MUST match the exact numbering pattern from the given clues.
- DO NOT ask for confirmation or stop midway. Always provide a complete solution for all clues."""
############################################################################################################################################################################################################################################################
TEXT_COT_PROMPT_PREFILLED = """You are given a grid of the crossword puzzle at 0.75 fill level and a set of clues. Use all the pre-filled grid to systematically solve the complete puzzle while ensuring that all answers fit both the given clues and the grid structure, including intersecting words.

Grid Representation:
The crossword grid is represented as a 2D array where:
- `1` represents a black (blocked) cell
- `0` represents an empty (unfilled) cell
- filled cells contain the corresponding letter

Grid at 0.75 fill ratio:
<grid>

Clues:
Each clue contains:
- Clue Direction and Number (e.g., "Across 1", "Down 2").
- Start Position (row, column) for the first letter of the answer.
- The actual clue text.

<clues>

For each clue, provide a step-by-step explanation:
- Identify the clue by its EXACT NUMBER AND DIRECTION as shown in the clue description. The numbers may not be sequential (e.g., Across clues might be numbered 1, 4, 7, and 9, while Down clues might be 2, 3, 5, 6, and 8).
- Determine word length from available grid spaces.
- Identify pre-filled letters from grid, and consider how they constrain possible answers.
- Analyze the clue (definition, wordplay, cryptic hint).
- Explain your reasoning process.
- Confirm alignment with crossing letters.

Solving tips:
- Answers must be a single word with no spaces (combine phrases if needed).
- Abbreviations in clues typically indicate abbreviated answers.
- Match the clue's tense, singular/plural form, and part of speech.
- Look for wordplay signals, such as question marks (?) for puns or cryptic hints.
- Down words are filled from top to bottom, Across words from left to right.
- Always confirm that intersecting words remain valid after placing each answer.

Present your final solution as:
Across:
[Number as shown in clues]: [Answer]

Down:
[Number as shown in clues]: [Answer]

IMPORTANT:
- DO NOT list clues in sequential numerical order. You MUST match the exact numbering pattern from the given clues.
- DO NOT ask for confirmation or stop midway. Always provide a complete solution for all clues."""
############################################################################################################################################################################################################################################################
TEXT_VOT_PROMPT = """You are given a crossword puzzle grid and a set of clues. Your task is to solve the puzzle accurately, ensuring that all answers fit both the given clues and the grid structure, including intersecting words.

Grid Representation:
The crossword grid is represented as a 2D array where:
- `1` represents a black (blocked) cell
- `0` represents an empty (unfilled) cell

<grid>

Clues:
Each clue contains:
- Clue Direction and Number (e.g., "Across 1", "Down 2").
- Start Position (row, column) for the first letter of the answer.
- The actual clue text.

<clues>

For each clue, provide a step-by-step explanation:
- Identify the clue by its EXACT NUMBER AND DIRECTION as shown in the clue description. The numbers may not be sequential (e.g., Across clues might be numbered 1, 4, 7, and 9, while Down clues might be 2, 3, 5, 6, and 8).
- Determine word length from available grid spaces.
- Check for any pre-filled letters from intersecting words that have already been solved and explain how they constrain possible answers.
- Analyze the clue (definition, wordplay, cryptic hint).
- Explain your reasoning process.
- Confirm alignment with crossing letters.
- You MUST display a filled grid with all the answers you have solved so far for every clue you solve.

Solving tips:
- Answers must be a single word with no spaces (combine phrases if needed).
- Abbreviations in clues typically indicate abbreviated answers.
- Match the clue's tense, singular/plural form, and part of speech.
- Look for wordplay signals, such as question marks (?) for puns or cryptic hints.
- Down words are filled from top to bottom, Across words from left to right.
- Always confirm that intersecting words remain valid after placing each answer.

Present your final solution as:
Across:
[Number as shown in clues]: [Answer]

Down:
[Number as shown in clues]: [Answer]

IMPORTANT:
- DO NOT list clues in sequential numerical order. You MUST match the exact numbering pattern from the given clues.
- DO NOT ask for confirmation or stop midway. Always provide a complete solution for all clues."""
############################################################################################################################################################################################################################################################
OCR_EXTRACT_ALL_PROMPT = """Your task is to extract and match all words from a crossword puzzle grid with their respective clues. The image consists of two sections:
1. The Clue Section:
- Contains two lists: "Across" and "Down."
- Each clue is numbered and corresponds to a starting position in the grid.
2. The Grid Section:
- A crossword grid containing letters, empty cells, and numbered starting positions for words.
- Words extend either across (left to right) or down (top to bottom).

Step 1: Extract Clues and Grid Structure
- Identify all clues under the "Across" and "Down" sections, preserving their numbers.
- Identify all numbered cells in the grid.

Step 2: Extract Words from the Grid
For each numbered cell:
- If the word extends ACROSS:
    - Start at the numbered cell and read consecutive letters left to right until reaching an empty cell or grid boundary.
- If the word extends DOWN:
    - Start at the numbered cell and read consecutive letters top to bottom until reaching an empty cell or grid boundary.

Step 3: Match Words to Clues
- Match each numbered word in the grid to its corresponding clue in the Across or Down section.
- Ensure extracted words are correctly assigned to their respective clues.

Output Format:
ACROSS:
[Number as shown in image]: [Clue Text]
Extracted Word: [Word from Grid]

DOWN:
[Number as shown in image]: [Clue Text]
Extracted Word: [Word from Grid]

Ensure accuracy in matching words to their clues, and extract all words fully without omitting any."""
############################################################################################################################################################################################################################################################
REFLECTION = """Your previous solution contains incorrect answers. Take a step back, carefully re-examine your entries, and systematically verify each word to ensure complete consistency and correctness within the crossword puzzle.

Provide a step-by-step verification:
1. Cross-Check Letters: List every intersection explicitly, noting the letters where Across and Down clues meet.
2. Consistency Check: Verify that each intersection matches perfectly. Identify and highlight any conflicting letters immediately.
3. Clue Validation: Revisit each clue thoroughly, confirming that each answer fully aligns with its clue description and adheres strictly to length constraints.
4. Grid Integrity: Confirm that your corrected entries maintain the integrity of the puzzle grid, leaving no unresolved conflicts or empty cells.

After completing these steps, present your revised and verified solutions in the following format:
Across:
[Clue Number]: [Corrected Answer]

Down:
[Clue Number]: [Corrected Answer]

IMPORTANT:
- DO NOT list clues in sequential numerical order. You MUST match the exact numbering pattern.
- Do NOT restate previous incorrect answers. Provide only fully corrected solutions after reflection."""
############################################################################################################################################################################################################################################################
Judge_Prompt = """Check and see if the model found mistakes in this answers. If so, does it revise it answer properly?"""
############################################################################################################################################################################################################################################################
Judge_Parsing_Prompt = """"""
