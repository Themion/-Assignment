package Parser;

import java.util.*;

public class Parser {
    // Recursive descent parser that inputs a C++Lite program and 
    // generates its abstract syntax.  Each method corresponds to
    // a concrete syntax grammar rule, which appears as a comment
    // at the beginning of the method.
  
    Token token;          // current token from the input stream
    Lexer lexer;
    static TreeNode root, handler;
    
    public Parser(Lexer ts) { // Open the C++Lite source program
        lexer = ts;                          // as a token stream, and
        token = lexer.next();            // retrieve its first Token
    }
  
    private String match (TokenType t) { // * return the string of a token if it matches with t *
        String value = token.value();
        if (token.type().equals(t))
            token = lexer.next();
        else if(t == TokenType.Semicolon) {
        	token = lexer.next();
        	match(t);
        }
        else
            error(t);
        return value;
    }
  
    private void error(TokenType tok) {
        error(tok.toString());
    }
  
    private void error(String tok) {
        System.err.println("Syntax error: expecting: " + tok 
                           + "; saw: " + token);
        root.print();
        System.exit(1);
    }
  
    public Program program() {
        // Program --> void main ( ) '{' Declarations Statements '}'
        TokenType[ ] header = {TokenType.Int, TokenType.Main,
                          TokenType.LeftParen, TokenType.RightParen};
        for (int i = 0; i<header.length; i++)   // bypass "int main ( )"
            match(header[i]);
        
        root = new TreeNode("int main(): ");
        handler = root;
        
        match(TokenType.LeftBrace);
        Declarations ds = declarations();
        Block ss = statements();
        match(TokenType.RightBrace);
        return new Program(ds, ss);  // student exercise
    }
  
    private Declarations declarations () {
        // Declarations --> { Declaration }
    	Declarations ds = new Declarations();
    	
    	TreeNode adder = new TreeNode("Declarations: ");
    	handler.addChild(adder);
    	handler = handler.get(handler.size() - 1);
    	
    	while(isType()) declaration(ds);
    	
    	handler = handler.getParent();
    	
		return ds;  // student exercise
    }
  
    private void declaration (Declarations ds) {
        // Declaration  --> Type Identifier { , Identifier } ;
    	Type t = type();
    	
    	
    	while(token.type() != TokenType.Eof) {
    		Variable v = new Variable(token.value());
    		match(TokenType.Identifier);

    		TreeNode adder = new TreeNode("Declaration: ");
    		adder.addChild("Type: " + t.toString());
    		ds.add(new Declaration(v, t));

    		adder.addChild("Variable: " + v.toString());
        	handler.addChild(adder);
    		
    		if(token.type() == TokenType.Comma)
				match(TokenType.Comma);
    		else if(token.type() == TokenType.Semicolon){
				match(TokenType.Semicolon);
				if(isType()) t = type();
				else break;
			}
    	}

        // student exercise
    }
  
    private Type type () {
        // Type  -->  int | bool | float | char 
        Type t = null;
        if(token.type() == TokenType.Int)
			t = Type.INT;
		else if(token.type() == TokenType.Bool)
			t = Type.BOOL;
		else if(token.type() == TokenType.Float)
			t = Type.FLOAT;
		else if(token.type() == TokenType.Char)
			t = Type.CHAR;
		else error("Current token not type (current token: " + token + ")");

		match(token.type());
        // student exercise
        return t;          
    }
  
    private Statement statement () {
        // Statement --> ; | Block | Assignment | IfStatement | WhileStatement
        Statement s = null;
        
        if(token.type() == TokenType.Semicolon)
        	s = new Skip();
        else if (token.type() == TokenType.If) 
	    	s = ifStatement();
	    else if (token.type() == TokenType.While)
	    	s = whileStatement();
	    else if (token.type() == TokenType.LeftBrace) {
			match(TokenType.LeftBrace);
			Block bl = statements();
			match(TokenType.RightBrace);
			
			s = bl;
	    }

		else if(token.type() == TokenType.Identifier){
			Variable name = new Variable(token.value());
			match(TokenType.Identifier);
			s = assignment(name);
		}
		else error("Unknown statement type: " + token.value());
        // student exercise
        return s;
    }
  
    private Block statements () {
        // Block --> '{' Statements '}'
    	Block b = new Block();
    	
    	TreeNode adder = new TreeNode("Statements: ");
    	handler.addChild(adder);
    	handler = handler.get(handler.size() - 1);
        
        while ((token.type() != TokenType.RightBrace) &&
        		(token.type() != TokenType.Eof))
        	b.members.add(statement());
        
        handler = handler.getParent();
    	
        return b;    // student exercise
    }
  
    private Assignment assignment (Variable i) {
        // Assignment --> Identifier = Expression ;

    	TreeNode adder = new TreeNode("=");
    	adder.addChild("Variable: " + i.toString());
    	adder.addChild("Value: ");
    	
    	handler.addChild(adder);
    	handler = handler.get(handler.size() - 1).get(1);
    	
    	match(TokenType.Assign);
		Expression e = expression();
    	match(TokenType.Semicolon);
    	
    	handler = handler.getParent().getParent();
    	
        return new Assignment(i, e);  // student exercise
    }
  
    private Conditional ifStatement () {
    	// IfStatement --> if ( Expression ) Statement [ else Statement ]
    	match(TokenType.If);

    	TreeNode adder = new TreeNode("If: ");
    	adder.addChild("Condition: ");
    	adder.addChild("Then: ");
    	
    	handler.addChild(adder);
    	handler = handler.get(handler.size() - 1).get(0);
    	
    	match(TokenType.LeftParen);
    	Expression e = expression();
    	match(TokenType.RightParen);
    	
    	handler = handler.getParent().get(1);
    	
    	Statement s = statement();
    	Statement el = new Skip();
    	
    	handler = handler.getParent();
    	
    	if(token.type() == TokenType.Else) {
    		token = lexer.next();
    		
    		handler.addChild("Else: ");
    		handler = handler.get(2);
    		el = statement();
    		handler = handler.getParent();
    	}
    	
		handler = handler.getParent();
    	
        return new Conditional(e, s, el);  // student exercise
    }
  
    private Loop whileStatement () {
        // WhileStatement --> while ( Expression ) Statement
    	match(TokenType.While);

    	TreeNode adder = new TreeNode("While: ");
    	adder.addChild("Condition: ");
    	adder.addChild("Then: ");
    	
    	handler.addChild(adder);
    	handler = handler.get(handler.size() - 1).get(0);
    	
    	match(TokenType.LeftParen);
    	Expression e = expression();
    	match(TokenType.RightParen);
    	
    	handler = handler.getParent().get(1);
    	
    	Statement s = statement();
    	
    	handler = handler.getParent().getParent();
    	
        return new Loop(e, s);  // student exercise
    }

    private Expression expression () {
        // Expression --> Conjunction { || Conjunction }
    	
    	handler.addChild("Expression: ");
    	handler = handler.get(handler.size() - 1);
    	
    	TreeNode adder;
    	
    	Expression e = conjunction();
    	
    	while (isEqualityOp()) {
    		Operator op = new Operator(match(token.type()));

        	adder = new TreeNode(op.toString());
        	adder.addChild(handler.get(0));
        	handler.remove(handler.size() - 1);
        	handler.addChild(adder);
        	handler = handler.get(handler.size() - 1);
        	
    		match(TokenType.Or);
        	Expression e2 = conjunction();
    		e = new Binary(op, e, e2);
            
            handler = handler.getParent();
    	}

        handler.popThis();
        handler = handler.getParent();
    	
        return e;  // student exercise
    }
  
    private Expression conjunction () {
        // Conjunction --> Equality { && Equality }
    	
    	handler.addChild("Conjunction: ");
    	handler = handler.get(handler.size() - 1);
    	
    	TreeNode adder;
    	
    	Expression c = equality();
    	
    	while (isEqualityOp()) {
    		Operator op = new Operator(match(token.type()));

        	adder = new TreeNode(op.toString());
        	adder.addChild(handler.get(0));
        	handler.remove(handler.size() - 1);
        	handler.addChild(adder);
        	handler = handler.get(handler.size() - 1);
        	
    		match(TokenType.And);
        	Expression c2 = equality();
    		c = new Binary(op, c, c2);
            
            handler = handler.getParent();
    	}

        handler.popThis();
        handler = handler.getParent();
        
    	return c;  // student exercise
    }
  
    private Expression equality () {
        // Equality --> Relation [ EquOp Relation ]
    	
    	handler.addChild("Equality: ");
    	handler = handler.get(handler.size() - 1);
    	
    	TreeNode adder;
    	
    	Expression e = relation();
    	
    	if (isEqualityOp()) {
    		Operator op = new Operator(match(token.type()));

        	adder = new TreeNode(op.toString());
        	adder.addChild(handler.get(0));
        	handler.remove(handler.size() - 1);
        	handler.addChild(adder);
        	handler = handler.get(handler.size() - 1);
    		
    		Expression e2 = relation();
    		e = new Binary(op, e, e2);
            
            handler = handler.getParent();
    	}

        handler.popThis();
        handler = handler.getParent();
    	
        return e;  // student exercise
    }

    private Expression relation (){
        // Relation --> Addition [RelOp Addition] 
    	
    	handler.addChild("Relation: ");
    	handler = handler.get(handler.size() - 1);
    	
    	TreeNode adder;
    	
    	Expression r = addition();
    	if (isRelationalOp()) {
    		Operator op = new Operator(match(token.type()));

        	adder = new TreeNode(op.toString());
        	adder.addChild(handler.get(0));
        	handler.remove(handler.size() - 1);
        	handler.addChild(adder);
        	handler = handler.get(handler.size() - 1);
    		
    		Expression r2 = addition();
    		r = new Binary(op, r, r2);
            
            handler = handler.getParent();
    	}

        handler.popThis();
        handler = handler.getParent();
    	
    	return r;  // student exercise
    }
  
    private Expression addition () {
        // Addition --> Term { AddOp Term }
    	
    	handler.addChild("Addition: ");
    	handler = handler.get(handler.size() - 1);
    	
    	TreeNode adder;
    	
        Expression e = term();
        
        while (isAddOp()) {
            Operator op = new Operator(match(token.type()));

        	adder = new TreeNode(op.toString());
        	adder.addChild(handler.get(0));
        	handler.remove(handler.size() - 1);
        	handler.addChild(adder);
        	handler = handler.get(handler.size() - 1);
    		
            Expression term2 = term();
            e = new Binary(op, e, term2);
            
            handler = handler.getParent();
        }

        handler.popThis();
        handler = handler.getParent();
        
        return e;
    }

    private Expression term () {
        // Term --> Factor { MultiplyOp Factor }
    	handler.addChild("Term: ");
    	handler = handler.get(handler.size() - 1);
    	
    	TreeNode adder;
    	
        Expression e = factor();
        
        while (isMultiplyOp()) {
            Operator op = new Operator(match(token.type()));

        	adder = new TreeNode(op.toString());
        	adder.addChild(handler.get(0));
        	handler.remove(handler.size() - 1);
        	handler.addChild(adder);
        	handler = handler.get(handler.size() - 1);
        	
            Expression term2 = factor();
            e = new Binary(op, e, term2);
            
            handler = handler.getParent();
        }

        handler.popThis();
        handler = handler.getParent();
        
        return e;
    }
  
    private Expression factor() {
        // Factor --> [ UnaryOp ] Primary 
        if (isUnaryOp()) {
            Operator op = new Operator(match(token.type()));
            
        	handler.addChild(op.toString());	//handler가 리프 노드라고 가정
        	handler = handler.get(0);
        	
            Expression term = primary();
            
            handler = handler.getParent();
            
            return new Unary(op, term);
        }
        else return primary();
    }
  
    private Expression primary () {
        // Primary --> Identifier | Literal | ( Expression )
        //             | Type ( Expression )
        Expression e = null;
        if (token.type().equals(TokenType.Identifier)) {
            e = new Variable(match(TokenType.Identifier));
            handler.addChild("Identifier: " + e.toString());
        } 
        else if (isLiteral()) {
            e = literal();
            handler.addChild("Literal: " + e.toString());
        } 
        else if (token.type().equals(TokenType.LeftParen)) {
        	handler.addChild("Paren: ");
        	handler = handler.get(handler.size() - 1);
        	
            token = lexer.next();
            e = expression();
            match(TokenType.RightParen);
            
            handler = handler.getParent();
        }
        else if (isType( )) {
            Operator op = new Operator(match(token.type()));

        	handler.addChild("Typecast: " + op.toString());
        	handler = handler.get(handler.size() - 1);
        	
            match(TokenType.LeftParen);
            Expression term = expression();
            match(TokenType.RightParen);
            e = new Unary(op, term);
            
            handler = handler.getParent();
        } 
        else error("Identifier | Literal | ( | Type");
        return e;
    }

    private Value literal( ) {
    	try{
			// int literal
			if (token.type() == TokenType.IntLiteral){
				Value v = new IntValue(Integer.parseInt(token.value()));
				match(TokenType.IntLiteral);
				return v;
				
			// float literal
			}else if (token.type() == TokenType.FloatLiteral){
				Value v = new FloatValue(Float.parseFloat(token.value()));
				match(TokenType.FloatLiteral);
				return v;
			}
			
			// char literal
			else if (token.type() == TokenType.CharLiteral){
				Value v = new CharValue(token.value().charAt(0));
				match(TokenType.CharLiteral);
				return v;
			}
			else
				error("unknown token type for literal! Token value: " + token.value());
		} catch(NumberFormatException e){
			error("Inavlid number format " + e.getLocalizedMessage());
		}
		return null;  // student exercise
    }
  

    private boolean isAddOp( ) {
        return token.type().equals(TokenType.Plus) ||
               token.type().equals(TokenType.Minus);
    }
    
    private boolean isMultiplyOp( ) {
        return token.type().equals(TokenType.Multiply) ||
               token.type().equals(TokenType.Divide);
    }
    
    private boolean isUnaryOp( ) {
        return token.type().equals(TokenType.Not) ||
               token.type().equals(TokenType.Minus);
    }
    
    private boolean isEqualityOp( ) {
        return token.type().equals(TokenType.Equals) ||
            token.type().equals(TokenType.NotEqual);
    }
    
    private boolean isRelationalOp( ) {
        return token.type().equals(TokenType.LessEqual) ||
               token.type().equals(TokenType.Less) || 
               token.type().equals(TokenType.GreaterEqual) ||
               token.type().equals(TokenType.Greater);
    }
    
    private boolean isType( ) {
        return token.type().equals(TokenType.Int)
            || token.type().equals(TokenType.Bool) 
            || token.type().equals(TokenType.Float)
            || token.type().equals(TokenType.Char);
    }
    
    private boolean isLiteral( ) {
        return token.type().equals(TokenType.IntLiteral) ||
            isBooleanLiteral() ||
            token.type().equals(TokenType.FloatLiteral) ||
            token.type().equals(TokenType.CharLiteral);
    }
    
    private boolean isBooleanLiteral( ) {
        return token.type().equals(TokenType.True) ||
            token.type().equals(TokenType.False);
    }
    
    public static void main(String args[]) {
    	System.out.print("Enter your clite file path: ");
    	Scanner scan = new Scanner(System.in);
    	String s = scan.nextLine();
        Parser parser  = new Parser(new Lexer(s));
        Program prog = parser.program();
        root.print();           // display abstract syntax tree
    } //main

} // Parser
