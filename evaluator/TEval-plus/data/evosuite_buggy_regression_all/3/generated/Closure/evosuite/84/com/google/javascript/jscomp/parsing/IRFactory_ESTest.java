/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 17:15:24 GMT 2023
 */

package com.google.javascript.jscomp.parsing;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.google.javascript.jscomp.mozilla.rhino.Context;
import com.google.javascript.jscomp.mozilla.rhino.ErrorReporter;
import com.google.javascript.jscomp.mozilla.rhino.Token;
import com.google.javascript.jscomp.mozilla.rhino.ast.ArrayComprehensionLoop;
import com.google.javascript.jscomp.mozilla.rhino.ast.ArrayLiteral;
import com.google.javascript.jscomp.mozilla.rhino.ast.AstRoot;
import com.google.javascript.jscomp.mozilla.rhino.ast.Block;
import com.google.javascript.jscomp.mozilla.rhino.ast.BreakStatement;
import com.google.javascript.jscomp.mozilla.rhino.ast.CatchClause;
import com.google.javascript.jscomp.mozilla.rhino.ast.Comment;
import com.google.javascript.jscomp.mozilla.rhino.ast.ConditionalExpression;
import com.google.javascript.jscomp.mozilla.rhino.ast.ContinueStatement;
import com.google.javascript.jscomp.mozilla.rhino.ast.DoLoop;
import com.google.javascript.jscomp.mozilla.rhino.ast.ElementGet;
import com.google.javascript.jscomp.mozilla.rhino.ast.EmptyExpression;
import com.google.javascript.jscomp.mozilla.rhino.ast.ErrorCollector;
import com.google.javascript.jscomp.mozilla.rhino.ast.ExpressionStatement;
import com.google.javascript.jscomp.mozilla.rhino.ast.ForLoop;
import com.google.javascript.jscomp.mozilla.rhino.ast.FunctionCall;
import com.google.javascript.jscomp.mozilla.rhino.ast.FunctionNode;
import com.google.javascript.jscomp.mozilla.rhino.ast.IfStatement;
import com.google.javascript.jscomp.mozilla.rhino.ast.Label;
import com.google.javascript.jscomp.mozilla.rhino.ast.LabeledStatement;
import com.google.javascript.jscomp.mozilla.rhino.ast.LetNode;
import com.google.javascript.jscomp.mozilla.rhino.ast.Name;
import com.google.javascript.jscomp.mozilla.rhino.ast.NewExpression;
import com.google.javascript.jscomp.mozilla.rhino.ast.NumberLiteral;
import com.google.javascript.jscomp.mozilla.rhino.ast.ObjectLiteral;
import com.google.javascript.jscomp.mozilla.rhino.ast.ObjectProperty;
import com.google.javascript.jscomp.mozilla.rhino.ast.ParenthesizedExpression;
import com.google.javascript.jscomp.mozilla.rhino.ast.PropertyGet;
import com.google.javascript.jscomp.mozilla.rhino.ast.RegExpLiteral;
import com.google.javascript.jscomp.mozilla.rhino.ast.ReturnStatement;
import com.google.javascript.jscomp.mozilla.rhino.ast.Scope;
import com.google.javascript.jscomp.mozilla.rhino.ast.StringLiteral;
import com.google.javascript.jscomp.mozilla.rhino.ast.SwitchCase;
import com.google.javascript.jscomp.mozilla.rhino.ast.ThrowStatement;
import com.google.javascript.jscomp.mozilla.rhino.ast.VariableDeclaration;
import com.google.javascript.jscomp.mozilla.rhino.ast.VariableInitializer;
import com.google.javascript.jscomp.mozilla.rhino.ast.WhileLoop;
import com.google.javascript.jscomp.mozilla.rhino.tools.ToolErrorReporter;
import com.google.javascript.jscomp.parsing.Config;
import com.google.javascript.jscomp.parsing.IRFactory;
import com.google.javascript.rhino.JSDocInfo;
import com.google.javascript.rhino.Node;
import java.util.LinkedHashSet;
import java.util.Set;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class IRFactory_ESTest extends IRFactory_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      EmptyExpression emptyExpression0 = new EmptyExpression();
      astRoot0.addChildToFront(emptyExpression0);
      ErrorCollector errorCollector0 = new ErrorCollector();
      // Undeclared exception!
      try { 
        IRFactory.transformTree(astRoot0, (String) null, (Config) null, errorCollector0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.parsing.IRFactory", e);
      }
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      ObjectProperty objectProperty0 = new ObjectProperty();
      astRoot0.addChildrenToFront(objectProperty0);
      ErrorCollector errorCollector0 = new ErrorCollector();
      // Undeclared exception!
      try { 
        IRFactory.transformTree(astRoot0, "/v%LsUq", (Config) null, errorCollector0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.parsing.IRFactory", e);
      }
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      DoLoop doLoop0 = new DoLoop();
      astRoot0.addChildToFront(doLoop0);
      ErrorCollector errorCollector0 = new ErrorCollector();
      // Undeclared exception!
      try { 
        IRFactory.transformTree(astRoot0, "language version", (Config) null, errorCollector0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.parsing.IRFactory", e);
      }
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      ElementGet elementGet0 = new ElementGet(23);
      astRoot0.addChildrenToFront(elementGet0);
      Context context0 = new Context();
      ErrorReporter errorReporter0 = context0.getErrorReporter();
      // Undeclared exception!
      try { 
        IRFactory.transformTree(astRoot0, "language version", (Config) null, errorReporter0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.parsing.IRFactory", e);
      }
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      Block block0 = new Block(19);
      astRoot0.addChildrenToFront(block0);
      ToolErrorReporter toolErrorReporter0 = new ToolErrorReporter(false);
      Node node0 = IRFactory.transformTree(astRoot0, " c!o|9p$_q3vv\"l1;", (Config) null, toolErrorReporter0);
      assertEquals(132, node0.getType());
      assertTrue(node0.hasOneChild());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      CatchClause catchClause0 = new CatchClause(8);
      Name name0 = new Name(0, "M\u0007UvOmX^(% ");
      catchClause0.setVarName(name0);
      astRoot0.addChildToFront(catchClause0);
      ErrorCollector errorCollector0 = new ErrorCollector();
      // Undeclared exception!
      try { 
        IRFactory.transformTree(astRoot0, "M\u0007UvOmX^(% ", (Config) null, errorCollector0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.parsing.IRFactory", e);
      }
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      WhileLoop whileLoop0 = new WhileLoop(9);
      astRoot0.addChildToFront(whileLoop0);
      ErrorCollector errorCollector0 = new ErrorCollector();
      // Undeclared exception!
      try { 
        IRFactory.transformTree(astRoot0, "error reporter", (Config) null, errorCollector0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.parsing.IRFactory", e);
      }
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      NumberLiteral numberLiteral0 = new NumberLiteral(573);
      astRoot0.addChildToFront(numberLiteral0);
      LinkedHashSet<String> linkedHashSet0 = new LinkedHashSet<String>();
      ErrorCollector errorCollector0 = new ErrorCollector();
      Config config0 = new Config(linkedHashSet0, linkedHashSet0, false, false);
      Node node0 = IRFactory.transformTree(astRoot0, "t~@,Yq< 2D);ZYF.z", config0, errorCollector0);
      assertEquals(132, node0.getType());
      assertEquals(1, node0.getChildCount());
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      ParenthesizedExpression parenthesizedExpression0 = new ParenthesizedExpression(12);
      astRoot0.addChildToFront(parenthesizedExpression0);
      ErrorCollector errorCollector0 = new ErrorCollector();
      // Undeclared exception!
      try { 
        IRFactory.transformTree(astRoot0, "error reporter", (Config) null, errorCollector0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.parsing.IRFactory", e);
      }
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      IfStatement ifStatement0 = new IfStatement(9, 9);
      ThrowStatement throwStatement0 = new ThrowStatement(ifStatement0);
      astRoot0.addChildToFront(throwStatement0);
      ErrorCollector errorCollector0 = new ErrorCollector();
      // Undeclared exception!
      try { 
        IRFactory.transformTree(astRoot0, "TMqwISFx", (Config) null, errorCollector0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.parsing.IRFactory", e);
      }
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      NewExpression newExpression0 = new NewExpression(22, 0);
      astRoot0.addChildrenToFront(newExpression0);
      ToolErrorReporter toolErrorReporter0 = new ToolErrorReporter(false);
      // Undeclared exception!
      try { 
        IRFactory.transformTree(astRoot0, "language version", (Config) null, toolErrorReporter0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.parsing.IRFactory", e);
      }
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      LinkedHashSet<String> linkedHashSet0 = new LinkedHashSet<String>();
      Config config0 = new Config(linkedHashSet0, linkedHashSet0, false, false);
      StringLiteral stringLiteral0 = new StringLiteral(4);
      astRoot0.addChild(stringLiteral0);
      ToolErrorReporter toolErrorReporter0 = new ToolErrorReporter(false);
      // Undeclared exception!
      try { 
        IRFactory.transformTree(astRoot0, "", config0, toolErrorReporter0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // StringNode: str is null
         //
         verifyException("com.google.javascript.rhino.Node$StringNode", e);
      }
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      Scope scope0 = new Scope(13, 6);
      astRoot0.addChildToFront(scope0);
      ErrorCollector errorCollector0 = new ErrorCollector();
      // Undeclared exception!
      try { 
        IRFactory.transformTree(astRoot0, (String) null, (Config) null, errorCollector0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.parsing.IRFactory", e);
      }
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      Label label0 = new Label(398, 5760, "_y+(K=S>fI");
      astRoot0.addChildrenToFront(label0);
      ToolErrorReporter toolErrorReporter0 = new ToolErrorReporter(false);
      Node node0 = IRFactory.transformTree(astRoot0, "YF", (Config) null, toolErrorReporter0);
      assertEquals(1, node0.getChildCount());
      assertEquals(132, node0.getType());
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      ExpressionStatement expressionStatement0 = new ExpressionStatement(astRoot0, true);
      astRoot0.addChildToFront(expressionStatement0);
      ErrorCollector errorCollector0 = new ErrorCollector();
      // Undeclared exception!
      try { 
        IRFactory.transformTree(astRoot0, "Unsupported syntax: ", (Config) null, errorCollector0);
        fail("Expecting exception: StackOverflowError");
      
      } catch(StackOverflowError e) {
         //
         // no message in exception (getMessage() returned null)
         //
      }
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      PropertyGet propertyGet0 = new PropertyGet();
      astRoot0.addChildToFront(propertyGet0);
      ErrorCollector errorCollector0 = new ErrorCollector();
      // Undeclared exception!
      try { 
        IRFactory.transformTree(astRoot0, "uto rage# dmensions", (Config) null, errorCollector0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.parsing.IRFactory", e);
      }
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      ArrayComprehensionLoop arrayComprehensionLoop0 = new ArrayComprehensionLoop(113, 2);
      astRoot0.addChildToFront(arrayComprehensionLoop0);
      ErrorCollector errorCollector0 = new ErrorCollector();
      // Undeclared exception!
      try { 
        IRFactory.transformTree(astRoot0, "rCY=A7ct.1mV\"aH8RJ", (Config) null, errorCollector0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.parsing.IRFactory", e);
      }
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      ForLoop forLoop0 = new ForLoop();
      astRoot0.addChildToFront(forLoop0);
      ErrorCollector errorCollector0 = new ErrorCollector();
      // Undeclared exception!
      try { 
        IRFactory.transformTree(astRoot0, "\"4?PWxZMz=I", (Config) null, errorCollector0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.parsing.IRFactory", e);
      }
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      JSDocInfo jSDocInfo0 = new JSDocInfo();
      Set<String> set0 = jSDocInfo0.getModifies();
      ConditionalExpression conditionalExpression0 = new ConditionalExpression(163, 12);
      astRoot0.addChildToFront(conditionalExpression0);
      Config config0 = new Config(set0, set0, true, false);
      ToolErrorReporter toolErrorReporter0 = new ToolErrorReporter(true);
      // Undeclared exception!
      try { 
        IRFactory.transformTree(astRoot0, "abId:}'etblB+V7)I$", config0, toolErrorReporter0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.parsing.IRFactory", e);
      }
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      LetNode letNode0 = new LetNode();
      astRoot0.addChildToFront(letNode0);
      ErrorCollector errorCollector0 = new ErrorCollector();
      // Undeclared exception!
      try { 
        IRFactory.transformTree(astRoot0, "language version", (Config) null, errorCollector0);
        fail("Expecting exception: UnsupportedOperationException");
      
      } catch(UnsupportedOperationException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.mozilla.rhino.ast.ErrorCollector", e);
      }
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      Token.CommentType token_CommentType0 = Token.CommentType.LINE;
      Comment comment0 = new Comment(13, 1, token_CommentType0, (String) null);
      astRoot0.addComment(comment0);
      ErrorCollector errorCollector0 = new ErrorCollector();
      Node node0 = IRFactory.transformTree(astRoot0, "P=[IBe{8v", (Config) null, errorCollector0);
      assertEquals(132, node0.getType());
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      LinkedHashSet<String> linkedHashSet0 = new LinkedHashSet<String>();
      Token.CommentType token_CommentType0 = Token.CommentType.BLOCK;
      Comment comment0 = new Comment(1, 20, token_CommentType0, "uto rage# dmensions");
      astRoot0.setJsDocNode(comment0);
      Config config0 = new Config(linkedHashSet0, linkedHashSet0, false, false);
      ToolErrorReporter toolErrorReporter0 = new ToolErrorReporter(false);
      Node node0 = IRFactory.transformTree(astRoot0, "uto rage# dmensions", config0, toolErrorReporter0);
      assertEquals(132, node0.getType());
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      ArrayLiteral arrayLiteral0 = new ArrayLiteral(2, 11);
      astRoot0.addChildToFront(arrayLiteral0);
      Node node0 = IRFactory.transformTree(astRoot0, "com.google.javascript.jscomp.parsing.IRFactory$1", (Config) null, (ErrorReporter) null);
      assertEquals(132, node0.getType());
      assertTrue(node0.hasChildren());
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      BreakStatement breakStatement0 = new BreakStatement(8, 8);
      astRoot0.addChildToFront(breakStatement0);
      ErrorCollector errorCollector0 = new ErrorCollector();
      Node node0 = IRFactory.transformTree(astRoot0, "P=[IBe{8v", (Config) null, errorCollector0);
      assertEquals(132, node0.getType());
      assertTrue(node0.hasOneChild());
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      CatchClause catchClause0 = new CatchClause(8);
      catchClause0.setCatchCondition(astRoot0);
      Name name0 = new Name(0, "M\u0007UvOmX^(% ");
      catchClause0.setVarName(name0);
      astRoot0.addChildToFront(catchClause0);
      ErrorCollector errorCollector0 = new ErrorCollector();
      // Undeclared exception!
      try { 
        IRFactory.transformTree(astRoot0, "M\u0007UvOmX^(% ", (Config) null, errorCollector0);
        fail("Expecting exception: UnsupportedOperationException");
      
      } catch(UnsupportedOperationException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.mozilla.rhino.ast.ErrorCollector", e);
      }
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      ContinueStatement continueStatement0 = new ContinueStatement();
      astRoot0.addChildToFront(continueStatement0);
      ErrorCollector errorCollector0 = new ErrorCollector();
      Node node0 = IRFactory.transformTree(astRoot0, "com.google.javascript.jscomp.mozilla.rhino.ast.RegExpLiteral", (Config) null, errorCollector0);
      assertEquals(1, node0.getChildCount());
      assertEquals(132, node0.getType());
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      Name name0 = new Name(21, 413);
      ContinueStatement continueStatement0 = new ContinueStatement(14, 9, name0);
      astRoot0.addChildToFront(continueStatement0);
      ErrorCollector errorCollector0 = new ErrorCollector();
      // Undeclared exception!
      try { 
        IRFactory.transformTree(astRoot0, "Tp4|4:>k]iO", (Config) null, errorCollector0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // StringNode: str is null
         //
         verifyException("com.google.javascript.rhino.Node$StringNode", e);
      }
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      FunctionNode functionNode0 = new FunctionNode();
      astRoot0.addChildToFront(functionNode0);
      ErrorCollector errorCollector0 = new ErrorCollector();
      // Undeclared exception!
      try { 
        IRFactory.transformTree(astRoot0, "9\"Kq%qBk4#0R_kB*2T9", (Config) null, errorCollector0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
      }
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      LabeledStatement labeledStatement0 = new LabeledStatement(1);
      astRoot0.addChildToFront(labeledStatement0);
      ErrorCollector errorCollector0 = new ErrorCollector();
      // Undeclared exception!
      try { 
        IRFactory.transformTree(astRoot0, "[~;pCynlsa'EV5!C6", (Config) null, errorCollector0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
      }
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      ObjectLiteral objectLiteral0 = new ObjectLiteral(13, 6);
      astRoot0.addChildToFront(objectLiteral0);
      ErrorCollector errorCollector0 = new ErrorCollector();
      Node node0 = IRFactory.transformTree(astRoot0, "com.google.javascript.jscomp.mozilla.rhino.tools.debugger.Dim$DimIProxy", (Config) null, errorCollector0);
      assertEquals(132, node0.getType());
      assertEquals(1, node0.getChildCount());
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      ObjectLiteral objectLiteral0 = new ObjectLiteral(13, 6);
      ObjectProperty objectProperty0 = new ObjectProperty(129, 11);
      objectLiteral0.addElement(objectProperty0);
      astRoot0.addChildToFront(objectLiteral0);
      ErrorCollector errorCollector0 = new ErrorCollector();
      // Undeclared exception!
      try { 
        IRFactory.transformTree(astRoot0, "com.google.javascript.jscomp.mozilla.rhino.tools.debugger.Dim$DimIProxy", (Config) null, errorCollector0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.parsing.IRFactory$TransformDispatcher", e);
      }
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      RegExpLiteral regExpLiteral0 = new RegExpLiteral();
      regExpLiteral0.setValue("!p<Z{|VDo");
      AstRoot astRoot0 = new AstRoot(23);
      astRoot0.addChildToFront(regExpLiteral0);
      Node node0 = IRFactory.transformTree(astRoot0, "!p<Z{|VDo", (Config) null, (ErrorReporter) null);
      assertEquals(132, node0.getType());
      assertEquals((-1), node0.getCharno());
      assertEquals(1, node0.getChildCount());
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      ReturnStatement returnStatement0 = new ReturnStatement(9);
      astRoot0.addChildToFront(returnStatement0);
      ErrorCollector errorCollector0 = new ErrorCollector();
      Node node0 = IRFactory.transformTree(astRoot0, "language version", (Config) null, errorCollector0);
      assertEquals(132, node0.getType());
      assertTrue(node0.hasChildren());
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      SwitchCase switchCase0 = new SwitchCase();
      SwitchCase switchCase1 = new SwitchCase();
      switchCase1.addStatement(switchCase0);
      astRoot0.addChildToFront(switchCase1);
      ErrorCollector errorCollector0 = new ErrorCollector();
      Node node0 = IRFactory.transformTree(astRoot0, "_!4Et(c6f.X<jo(dJ{", (Config) null, errorCollector0);
      assertEquals(132, node0.getType());
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      VariableDeclaration variableDeclaration0 = new VariableDeclaration(2, 0);
      astRoot0.addChildToFront(variableDeclaration0);
      ErrorCollector errorCollector0 = new ErrorCollector();
      Node node0 = IRFactory.transformTree(astRoot0, "zYHgsN1:u}", (Config) null, errorCollector0);
      assertEquals(132, node0.getType());
  }

  @Test(timeout = 4000)
  public void test35()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      VariableDeclaration variableDeclaration0 = new VariableDeclaration(2, 0);
      VariableInitializer variableInitializer0 = new VariableInitializer(14, 1183);
      variableDeclaration0.addVariable(variableInitializer0);
      astRoot0.addChildToFront(variableDeclaration0);
      ErrorCollector errorCollector0 = new ErrorCollector();
      // Undeclared exception!
      try { 
        IRFactory.transformTree(astRoot0, "~:; ]KHj/m,gTsMF", (Config) null, errorCollector0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
      }
  }

  @Test(timeout = 4000)
  public void test36()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      FunctionCall functionCall0 = new FunctionCall();
      astRoot0.addChildToFront(functionCall0);
      ErrorCollector errorCollector0 = new ErrorCollector();
      // Undeclared exception!
      try { 
        IRFactory.transformTree(astRoot0, "language version", (Config) null, errorCollector0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
      }
  }

  @Test(timeout = 4000)
  public void test37()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      ExpressionStatement expressionStatement0 = new ExpressionStatement(0, 1);
      astRoot0.addChildrenToFront(expressionStatement0);
      Context context0 = new Context();
      ErrorReporter errorReporter0 = context0.getErrorReporter();
      // Undeclared exception!
      try { 
        IRFactory.transformTree(astRoot0, "language version", (Config) null, errorReporter0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
      }
  }
}