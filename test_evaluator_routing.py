#!/usr/bin/env python3
"""
Test script to verify evaluator routing works correctly for both MCQ and arithmetic datasets.
This tests that the get_evaluator() function correctly routes to the appropriate evaluator.
"""

import sys
import os

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
sys.path.insert(0, os.path.dirname(__file__))

from evaluator import (
    evaluate_arithmetics,
    evaluate_mcq,
    base_evaluate_arithmetics,
    base_evaluate_mcq,
    get_instruction_suffix
)


def test_evaluators():
    """Test that evaluators work correctly"""
    
    print("=" * 60)
    print("TESTING EVALUATORS")
    print("=" * 60)
    
    # Test MCQ evaluator
    print("\n1. Testing MCQ evaluator with correct answer...")
    test_responses_mcq = {'agent_1': 'The answer is {final answer: (C)}'}
    final_answers, debate_answer, is_correct = evaluate_mcq(test_responses_mcq, '(C)')
    print(f"   Final answers: {final_answers}")
    print(f"   Debate answer: {debate_answer}")
    print(f"   Is correct: {is_correct}")
    assert is_correct == True, "MCQ should be correct"
    assert debate_answer == '(C)', f"Expected (C), got {debate_answer}"
    print("   PASS")
    
    print("\n2. Testing MCQ evaluator with incorrect answer...")
    final_answers, debate_answer, is_correct = evaluate_mcq(test_responses_mcq, '(D)')
    print(f"   Final answers: {final_answers}")
    print(f"   Debate answer: {debate_answer}")
    print(f"   Is correct: {is_correct}")
    assert is_correct == False, "MCQ should be incorrect"
    print("   PASS")
    
    # Test Arithmetic evaluator
    print("\n3. Testing arithmetic evaluator with correct answer...")
    test_responses_arith = {'agent_1': 'The answer is {final answer: 42.5}'}
    final_answers, debate_answer, is_correct = evaluate_arithmetics(test_responses_arith, 42.5)
    print(f"   Final answers: {final_answers}")
    print(f"   Debate answer: {debate_answer}")
    print(f"   Is correct: {is_correct}")
    assert is_correct == True, "Arithmetic should be correct"
    assert debate_answer == 42.5, f"Expected 42.5, got {debate_answer}"
    print("   PASS")
    
    print("\n4. Testing arithmetic evaluator with incorrect answer...")
    final_answers, debate_answer, is_correct = evaluate_arithmetics(test_responses_arith, 40.0)
    print(f"   Final answers: {final_answers}")
    print(f"   Debate answer: {debate_answer}")
    print(f"   Is correct: {is_correct}")
    assert is_correct == False, "Arithmetic should be incorrect"
    print("   PASS")


def test_instruction_suffix():
    """Test that instruction suffix is correct for each dataset"""
    
    print("\n" + "=" * 60)
    print("TESTING INSTRUCTION SUFFIXES")
    print("=" * 60)
    
    class Args:
        bae = False
        cot = False
    
    print("\n1. GSM8K suffix (arithmetic)...")
    args = Args()
    args.data = 'gsm8k'
    suffix = get_instruction_suffix(args)
    print(f"   Suffix: {suffix[:60]}...")
    assert 'final answer: 123' in suffix.lower(), "Should mention numeric format"
    print("   PASS")
    
    print("\n2. Pro Medicine suffix (MCQ)...")
    args.data = 'pro_medicine'
    suffix = get_instruction_suffix(args)
    print(f"   Suffix: {suffix[:60]}...")
    assert 'final answer: (a)' in suffix.lower(), "Should mention MCQ format"
    print("   PASS")
    
    print("\n3. Formal Logic suffix (MCQ)...")
    args.data = 'formal_logic'
    suffix = get_instruction_suffix(args)
    print(f"   Suffix: {suffix[:60]}...")
    assert 'final answer: (a)' in suffix.lower(), "Should mention MCQ format"
    print("   PASS")


def test_router():
    """Test the router function"""
    
    print("\n" + "=" * 60)
    print("TESTING EVALUATOR ROUTER (get_evaluator)")
    print("=" * 60)
    
    from src.main import get_evaluator
    
    class Args:
        bae = False
    
    print("\n1. Router for GSM8K...")
    args = Args()
    args.data = 'gsm8k'
    evaluator = get_evaluator(args)
    print(f"   Selected evaluator: {evaluator.__name__}")
    assert evaluator == evaluate_arithmetics, f"Expected evaluate_arithmetics, got {evaluator.__name__}"
    print("   PASS")
    
    print("\n2. Router for Pro Medicine...")
    args.data = 'pro_medicine'
    evaluator = get_evaluator(args)
    print(f"   Selected evaluator: {evaluator.__name__}")
    assert evaluator == evaluate_mcq, f"Expected evaluate_mcq, got {evaluator.__name__}"
    print("   PASS")
    
    print("\n3. Router for Formal Logic...")
    args.data = 'formal_logic'
    evaluator = get_evaluator(args)
    print(f"   Selected evaluator: {evaluator.__name__}")
    assert evaluator == evaluate_mcq, f"Expected evaluate_mcq, got {evaluator.__name__}"
    print("   PASS")
    
    print("\n4. Router with BAE flag for GSM8K...")
    args = Args()
    args.data = 'gsm8k'
    args.bae = True
    evaluator = get_evaluator(args)
    print(f"   Selected evaluator: {evaluator.__name__}")
    assert evaluator == base_evaluate_arithmetics, f"Expected base_evaluate_arithmetics, got {evaluator.__name__}"
    print("   PASS")
    
    print("\n5. Router with BAE flag for Pro Medicine...")
    args.data = 'pro_medicine'
    evaluator = get_evaluator(args)
    print(f"   Selected evaluator: {evaluator.__name__}")
    assert evaluator == base_evaluate_mcq, f"Expected base_evaluate_mcq, got {evaluator.__name__}"
    print("   PASS")


if __name__ == '__main__':
    try:
        test_evaluators()
        test_instruction_suffix()
        test_router()
        
        print("\n" + "=" * 60)
        print("SUCCESS: All evaluator routing tests passed!")
        print("=" * 60)
        print("\nThe evaluator routing is now working correctly.")
        print("You can safely run with:")
        print("  - python src/main.py --model qwen2.5-7b --data gsm8k ...")
        print("  - python src/main.py --model qwen2.5-7b --data pro_medicine ...")
        print("  - python src/main.py --model qwen2.5-7b --data formal_logic ...")
        
    except Exception as e:
        print(f"\n[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
