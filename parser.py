from lark import Lark

def parse_file(filename):
    """Parses the given file, producing the corresponding directives."""
    l = Lark('''
            %import common.NUMBER   
            %ignore " "
            %ignore "\\n"
            
            ?start: program
            program: from_data train_a predicting write_model_to protected_classes required_fairness explanation
            
            ?string: "\\"" /[^"]+/ "\\""
            
            from_data: _FROM_DATA string 
            train_a: _TRAIN_A string 
            predicting: _PREDICTING string 
            write_model_to: _WRITE_MODEL_TO string 
            protected_classes: _PROTECTED_CLASSES string ( "," string )*
            required_fairness: _REQUIRED_FAIRNESS "(" "disparate impact <" /[0-9]+.[0-9]+/ ")" 
            explanation: _EXPLANATION string  

            _FROM_DATA: "FROM DATA"  
            _TRAIN_A: "TRAIN A"  
            _PREDICTING: "PREDICTING"  
            _WRITE_MODEL_TO: "WRITE MODEL TO"  
            _PROTECTED_CLASSES: "PROTECTED CLASSES"  
            _REQUIRED_FAIRNESS: "REQUIRED FAIRNESS"  
            _EXPLANATION: "EXPLANATION"  
            _AGGREGATE: "AGGREGATE"  

         ''')

    with open(filename, 'r') as f:
        tree = l.parse(f.read())

    directives = {}
    for directive_name in [
        "from_data",
        "train_a",
        "predicting",
        "write_model_to",
        "protected_classes",
        "required_fairness",
        "explanation",
        ]:
        directives[directive_name] = str(next(tree.find_data(directive_name)).children[0])

    return directives