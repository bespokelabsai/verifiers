from typing import List, Dict, Any
import ast
import json
import time
from verifiers.parsers import XMLParser
from verifiers.rubrics import Rubric

class BfclRubric(Rubric):
    def __init__(self,
                 parser: XMLParser = XMLParser(fields=["reasoning", "tool"]),
                 env_parser: XMLParser = XMLParser(fields=["tool_result"])):
        self.parser = parser
        self.env_parser = env_parser
        self.reward_funcs = [
            self.tool_execution_reward_func,
            self.unified_reward_func,
        ]

    def tool_execution_reward_func(self, completions: List[List[Dict[str, str]]], states: List[Dict[str, Any]], 
                                   debug: bool = False, max_score: float = 0.2) -> List[float]:
        """
        Reward function that checks tool execution success.

        Uses XMLParser to identify proper tool calls.
        """
        if debug:
            print(f"Computing Tool Execution Reward\n")
            time.sleep(3)
        def check_execution(trajectory, debug=debug):
            tool_attempts = 0
            successful_executions = 0
            
            # Find assistant messages with tools and their responses
            for i, msg in enumerate(trajectory):
                if msg['role'] == 'assistant':
                    if debug:
                        print(f"LLM Response: {msg['content']}\n")
                        time.sleep(3)
                    # Use parser to check for tool tag
                    parsed = self.parser.parse(msg['content'])
                    if hasattr(parsed, 'tool') and parsed.tool is not None:
                        # Found a properly formatted tool message
                        if i + 1 < len(trajectory) and trajectory[i + 1]['role'] == 'system':
                            if debug:
                                print(f"Found properly formatted tool message: {parsed.tool}\n")
                                time.sleep(3)
                            # Check response with env_parser
                            parsed_response = self.env_parser.parse(trajectory[i + 1]['content'])
                            if hasattr(parsed_response, 'tool_result') and parsed_response.tool_result is not None:
                                try:
                                    tool_results = json.loads(parsed_response.tool_result)
                                except json.JSONDecodeError as e: # NOTE: This means parser malfunctions due to potentially double tags
                                    tool_results = []
                                except Exception as e:
                                    print(f"Tool Result: {parsed_response.tool_result}")
                                    raise Exception(f"Error in Parsing Tool Result is Not Expected!! Error: {e}")
                                for tool_result in tool_results:
                                    tool_attempts += 1
                                    if not "error" in tool_result.lower():
                                        successful_executions += 1
                                        if debug:
                                            print(f"Successful execution: {tool_result}\n")
                                            time.sleep(3)
                                    else:
                                        if debug:
                                            print(f"Error in execution: {tool_result}\n")
                                            time.sleep(3)

                            # if hasattr(parsed_response, 'tool_result') and parsed_response.tool_result is not None and not "error" in parsed_response.tool_result.lower():
                            #     successful_executions += 1
                            #     if debug:
                            #         print(f"Successful execution: {parsed_response.tool_result}")
                            #         time.sleep(3)
                            # else:
                            #     successful_executions += 0
                            #     if debug:
                            #         print(f"Error in execution: {parsed_response.tool_result}")
                            #         time.sleep(3)
            if debug:
                print(f"Successful executions: {successful_executions}")
                print(f"Tool attempts: {tool_attempts}\n")
                time.sleep(3)
            # Calculate reward
            if tool_attempts == 0:
                if debug:
                    print(f"Found no tool calls in the trajectory\n")
                    time.sleep(3)
                return 0.0
            final_score = max_score * (successful_executions / tool_attempts) if tool_attempts > 0 else 0.0
            if debug:
                print(f"Final Tool Execution Score: {final_score}\n")
                time.sleep(3)
            return final_score
        
        return [check_execution(c, debug=(debug and (j==0))) for j, c in enumerate(completions)]

    @staticmethod
    def _parse_function_call(func_call_str: str):
        """
        Parses a function call string into a JSON-like dictionary.
        
        :param func_call_str: String representation of a function call.
        :return: JSON-like dictionary with function name and arguments.
        """
        try:
            # Parse the function call string into an AST node
            tree = ast.parse(func_call_str, mode='eval')

            # Ensure it is a function call
            if not isinstance(tree.body, ast.Call):
                raise ValueError("Input is not a valid function call.")

            # Extract function name
            func_name = tree.body.func.id if isinstance(tree.body.func, ast.Name) else None
            if not func_name:
                raise ValueError("Could not determine function name.")

            # Extract arguments
            args_dict = {}

            # Handle keyword arguments (named parameters)
            for kw in tree.body.keywords:
                args_dict[kw.arg] = ast.literal_eval(kw.value)  # Convert AST to actual Python value

            # Handle positional arguments (if any, though your example has none)
            for i, arg in enumerate(tree.body.args):
                args_dict[f"arg{i+1}"] = ast.literal_eval(arg)

            # Create JSON output
            json_obj = {
                "name": func_name,
                "args": args_dict
            }

            return json_obj

        except Exception:
            raise Exception(f"Error in Parsing Ground Truth Function Call is Not Expected!!")

    @staticmethod
    def _is_subsequence_unordered(list1, list2) -> tuple[bool, list]:
        """
        Checks if all elements of list1 are present in list2, regardless of order.
        Also returns the elements of list1 that are not present in list2.
        """
        if list1 == [] or list2 == []:
            return False, []
        # Copy list2 to avoid modifying the original list during checks
        list2_copy = list2[:]
        
        # Check each item in list1 to see if it exists in list2_copy
        missing_elements = []
        for item in list1:
            try:
                # Attempt to remove one occurrence of `item` from list2_copy to handle duplicates
                list2_copy.remove(item)
            except ValueError:
                # If item is not found, add it to missing_elements
                missing_elements.append(item)
        
        # If there are missing elements, list1 is not a subsequence of list2
        is_subsequence = len(missing_elements) == 0
        return is_subsequence, missing_elements

    @staticmethod
    def compare_instances(model_obect, ground_truth_object):
        """
        Checks if the model_object has the same attributes as the ground_truth_object. They are instances of the same class.
        """
        assert type(model_obect) == type(
            ground_truth_object
        ), "Objects are not of the same type."
        differences = {}
        valid = True
        for attr_name in vars(ground_truth_object):
            # We don't check for private attributes
            if attr_name.startswith("_"):
                continue
            model_attr = getattr(model_obect, attr_name)
            ground_truth_attr = getattr(ground_truth_object, attr_name)

            if model_attr != ground_truth_attr:
                valid = False
                differences[attr_name] = {"model": model_attr, "ground_truth": ground_truth_attr}

        return valid, differences
    
    def unified_reward_func(self, completions: List[List[Dict[str, str]]], states: List[Dict[str, Any]], 
                          debug: bool = False, 
                          func_match_max_score: float = 0.5, state_match_max_score: float = 0.5, 
                          format_max_score: float = 0.2) -> List[float]:
        """
        Combined reward function that checks state matches, function call matches, and format.
        State and function matches contribute 0.5 each to base score.
        If base score is perfect, format check can add 0.2 more.
        """
        if debug:
            print(f"Computing Unified Reward\n")
            time.sleep(3)

        def check_unified(trajectory, state, debug=debug):
            # First check state matches
            if debug:
                print(f"Checking State Matches\n")
                time.sleep(3)
            num_state_matches = 0
            num_state_total = 0
            for key in state["ground_truth_environment"]:
                if debug:
                    print(f"Comparing {key} in ground truth and environment")
                    print("Current Environment Attributes:")
                    for attr_name, value in vars(state['environment'][key]).items():
                        if not attr_name.startswith('_'):
                            print(f"  {attr_name}: {value}")
                    print("\nGround Truth Environment Attributes:")
                    for attr_name, value in vars(state['ground_truth_environment'][key]).items():
                        if not attr_name.startswith('_'):
                            print(f"  {attr_name}: {value}")
                    time.sleep(3)

                valid, diffs = self.compare_instances(state["ground_truth_environment"][key], state["environment"][key])
                if debug:
                    print(f"State Match: {valid}")
                    print(f"Differences: {diffs}")
                    time.sleep(3)
                num_state_matches += int(valid)
                num_state_total += 1
                
            state_score = state_match_max_score * (num_state_matches / num_state_total)
            if debug:
                print(f"State Score: {state_score}\n")
                time.sleep(3)

            # Then check function calls
            if debug:
                print(f"Checking Function Calls\n")
                time.sleep(3)
            num_func_matches = 0
            num_func_total = 0
            model_func_calls = state["successful_func_calls"]
            ground_truth_func_calls = json.loads(state['dataset_row']['answer'])
            assert len(model_func_calls) == len(ground_truth_func_calls)

            for model_calls, gt_calls_str in zip(model_func_calls, ground_truth_func_calls):
                gt_calls = [self._parse_function_call(call_str) for call_str in gt_calls_str]
                
                def make_hashable(value):
                    if isinstance(value, dict):
                        return frozenset((k, make_hashable(v)) for k, v in value.items())
                    elif isinstance(value, list):
                        return tuple(make_hashable(item) for item in value)
                    elif isinstance(value, set):
                        return frozenset(make_hashable(item) for item in value)
                    return value

                comparable_model_calls = [
                    (call["name"], frozenset((k, make_hashable(v)) for k, v in call["args"].items()))
                    for call in model_calls
                ]
                
                for call in gt_calls:
                    if "args" in call:
                        for key, value in call["args"].items():
                            if isinstance(value, list):
                                call["args"][key] = tuple(value)
                    else:
                        raise Exception("Error in Parsing Ground Truth Function Call is Not Expected!!")

                comparable_gt_calls = [
                    (call["name"], frozenset((k, make_hashable(v)) for k, v in call["args"].items()))
                    for call in gt_calls
                ]
                if debug:
                    print(f"Comparable Model Calls: {comparable_model_calls}")
                    print(f"Comparable Ground Truth Calls: {comparable_gt_calls}")
                    time.sleep(3)

                is_match, _ = self._is_subsequence_unordered(comparable_gt_calls, comparable_model_calls)
                if debug:
                    print(f"Is Subsequence: {is_match}")
                    time.sleep(3)
                num_func_matches += int(is_match)
                num_func_total += 1
            func_score = func_match_max_score * (num_func_matches / num_func_total)
            if debug:
                print(f"Function Call Score: {func_score}\n")
                time.sleep(3)

            base_score = state_score + func_score
            if base_score != state_match_max_score + func_match_max_score:
                if debug:
                    print(f"Base Score is not perfect, so giving 0 score, and no format check.\n")
                    time.sleep(3)
                format_score = 0
                base_score = 0
            # Only check format if base score is perfect
            else:
                if debug:
                    print(f"Base Score is perfect, checking format\n")
                    time.sleep(3)
                valid_messages = 0
                total_messages = 0
                for msg in trajectory:
                    if msg['role'] == 'assistant':
                        if debug:
                            print(f"Checking Message: {msg['content']}")
                            time.sleep(3)
                        total_messages += 1
                        parsed = self.parser.parse(msg['content'])
                        if debug:
                            print(f"Parsed: {parsed}")
                            time.sleep(3)
                        # Must have reasoning content
                        if not hasattr(parsed, 'reasoning') or parsed.reasoning is None:
                            if debug:
                                print(f"Valid: False")
                                time.sleep(3)
                            continue
                        
                        # Must have either tool content or task status
                        if (hasattr(parsed, 'tool') and parsed.tool is not None) or ("<TASK_FINISHED>" in msg['content'] or "<TASK_ERROR>" in msg['content']):
                            valid_messages += 1
                            if debug:
                                print(f"Valid: True")
                                time.sleep(3)
                        else:
                            if debug:
                                print(f"Valid: False")
                                time.sleep(3)

                format_score = format_max_score * (valid_messages / total_messages) if total_messages > 0 else 0
            base_score += format_score

            # if debug:
            #     print(f"State Score: {state_score}")
            #     print(f"Function Call Score: {func_score}")
            #     print(f"Format Score: {format_score}")
            #     print(f"Final Unified Score: {base_score}")
            #     time.sleep(3)
            return base_score

        return [check_unified(c, s, debug=(debug and (j==0))) for j, (c, s) in enumerate(zip(completions, states))]