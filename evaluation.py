# evaluation.py
import re

def evaluate_exam_paper(text: str) -> dict:
    """
    Evaluate generated exam text on various criteria with simple heuristics.
    Returns scores from 1-5 for each metric.
    """
    # Relevance: presence of Grade 7 and typical exam keywords
    relevance_score = 5 if ("Grade 7" in text and 
                            any(keyword in text for keyword in ["Comprehension", "Multiple Choice", "Section"])) else 3
    
    # Coverage: number of questions detected
    question_count = len(re.findall(r"\bQuestion\b|\bQ\d+\.|\b\d+\.", text))
    coverage_score = 5 if question_count >= 40 else max(1, int(question_count / 10))
    
    # Complexity: average length of questions (approximate complexity)
    questions = re.split(r"\bQuestion\b|\bQ\d+\.|\b\d+\.", text)
    avg_length = sum(len(q) for q in questions) / max(1, len(questions))
    complexity_score = 5 if avg_length > 80 else 3
    
    # Structure Quality: presence of clear exam sections
    has_instructions = "===INSTRUCTIONS===" in text
    has_questions = "===QUESTIONS===" in text
    structure_score = 5 if has_instructions and has_questions else 3
    
    # Redundancy: estimated by ratio of unique lines to total lines
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    unique_lines = set(lines)
    redundancy_ratio = len(unique_lines) / max(1, len(lines))
    redundancy_score = 5 if redundancy_ratio > 0.9 else 2

    return {
        "Relevance": relevance_score,
        "Coverage": coverage_score,
        "Complexity": complexity_score,
        "Structure Quality": structure_score,
        "Redundancy": redundancy_score
    }
