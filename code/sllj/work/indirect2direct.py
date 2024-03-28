import re
import regex


class ProofSystem:
    def __init__(self, gamma, known_proof, gamma_set=None, additional_axioms=None):
        self.gamma = gamma
        self.known_proof = known_proof
        self.gamma_set = gamma_set if gamma_set else []
        self.additional_axioms = additional_axioms if additional_axioms else []

    def find_and_remove_substring(self, input_string, substring):
        index = input_string.find(substring)
        if index != -1:
            return input_string[:index] + input_string[index+len(substring):], index
        else:
            return "Substring not found in the input string."

    def is_arrow_concat(self, expression):
        """Check if the expression is a concatenation of two expressions with an arrow."""
        pattern = r'\((?>[^()]|(?R))*\)'
        matchs = regex.findall(pattern, expression)
        if len(matchs) == 2:
            return True
        elif len(matchs) == 1:
            if '->' in self.find_and_remove_substring(expression, matchs[0])[0]:
                return True
        else:
            if '->' in expression:
                return True
        return False

    def is_need_pth(self, expression):
        """Check if the expression needs parentheses."""
        pattern = r'~+\(.*\)'
        if '->' not in expression:
            return False
        elif re.match(pattern, expression):
            return False
        else:
            return True

    # Other methods...

    def main(self):
        '''Main function'''
        alpha = self.process_alpha()

        if self.is_arrow_concat(self.gamma):
            A, B = self.extract_expression_by_arrow(self.gamma)
        else:
            print("The input formula is not in the correct format.")
            return
        proof_set = []
        indexs = []
        rules = []
        self.add_to_proof_set(proof_set, A, alpha, self.gamma_set,
                              (len(self.known_proof)-1, B), indexs, rules)

        indexs = self.inverse_permutation(indexs)

        for i in range(len(rules)):
            if 'MP' in rules[i]:
                nums = re.findall(r'\d+', rules[i])
                for num in nums:
                    rules[i] = re.sub(r'{}'.format(num), r'{}'.format(
                        indexs[int(num)-1]), rules[i])

        # Write to file or print...

    # Other methods...

    def process_alpha(self):
        """Process the set of axioms."""
        alpha = []
        for axiom in self.additional_axioms:
            processed_axiom = self.replace_string_with_regex(axiom)
            alpha.append(processed_axiom)
        return alpha


# Example usage:
gamma_input = input("Enter the formula to be proved: ")
known_proof_input = input("Enter the deductive theorem proof: ").split()
gamma_set_input = input("Enter the known set Gamma, if any: ").split()
additional_axioms_input = input(
    "Enter any additional axioms, if any: ").split()

proof_system = ProofSystem(
    gamma_input, known_proof_input, gamma_set_input, additional_axioms_input)
proof_system.main()
