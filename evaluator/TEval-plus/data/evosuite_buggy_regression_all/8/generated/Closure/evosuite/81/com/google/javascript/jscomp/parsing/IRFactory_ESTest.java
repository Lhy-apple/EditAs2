/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 02:17:35 GMT 2023
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
import com.google.javascript.jscomp.mozilla.rhino.ast.IfStatement;
import com.google.javascript.jscomp.mozilla.rhino.ast.Name;
import com.google.javascript.jscomp.mozilla.rhino.ast.NewExpression;
import com.google.javascript.jscomp.mozilla.rhino.ast.NumberLiteral;
import com.google.javascript.jscomp.mozilla.rhino.ast.ObjectLiteral;
import com.google.javascript.jscomp.mozilla.rhino.ast.ReturnStatement;
import com.google.javascript.jscomp.mozilla.rhino.ast.Scope;
import com.google.javascript.jscomp.mozilla.rhino.ast.StringLiteral;
import com.google.javascript.jscomp.mozilla.rhino.ast.SwitchCase;
import com.google.javascript.jscomp.mozilla.rhino.ast.ThrowStatement;
import com.google.javascript.jscomp.mozilla.rhino.ast.WhileLoop;
import com.google.javascript.jscomp.mozilla.rhino.ast.WithStatement;
import com.google.javascript.jscomp.mozilla.rhino.tools.ToolErrorReporter;
import com.google.javascript.jscomp.parsing.Config;
import com.google.javascript.jscomp.parsing.IRFactory;
import com.google.javascript.rhino.Node;
import java.nio.charset.Charset;
import java.util.LinkedHashSet;
import java.util.Set;
import java.util.TreeSet;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.mock.java.io.MockFile;
import org.evosuite.runtime.mock.java.io.MockPrintStream;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class IRFactory_ESTest extends IRFactory_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      LinkedHashSet<String> linkedHashSet0 = new LinkedHashSet<String>();
      EmptyExpression emptyExpression0 = new EmptyExpression();
      astRoot0.addChild(emptyExpression0);
      Config config0 = new Config(linkedHashSet0, linkedHashSet0, true, true, true);
      Node node0 = IRFactory.transformTree(astRoot0, "", config0, (ErrorReporter) null);
      assertTrue(node0.hasChildren());
      assertEquals(132, node0.getType());
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      DoLoop doLoop0 = new DoLoop(4, 2);
      astRoot0.addChild(doLoop0);
      ToolErrorReporter toolErrorReporter0 = new ToolErrorReporter(true);
      // Undeclared exception!
      try { 
        IRFactory.transformTree(astRoot0, "", (Config) null, toolErrorReporter0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
      }
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      ElementGet elementGet0 = new ElementGet(0);
      astRoot0.addChild(elementGet0);
      ErrorCollector errorCollector0 = new ErrorCollector();
      // Undeclared exception!
      try { 
        IRFactory.transformTree(astRoot0, "#lCWoOY<^jz", (Config) null, errorCollector0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
      }
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      LinkedHashSet<String> linkedHashSet0 = new LinkedHashSet<String>();
      Name name0 = new Name((-563), 2);
      ContinueStatement continueStatement0 = new ContinueStatement(5, 2, name0);
      astRoot0.addChildToFront(continueStatement0);
      Config config0 = new Config(linkedHashSet0, linkedHashSet0, true, true, true);
      ToolErrorReporter toolErrorReporter0 = new ToolErrorReporter(true);
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
  public void test04()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      WhileLoop whileLoop0 = new WhileLoop((-3322));
      astRoot0.addChild(whileLoop0);
      ErrorCollector errorCollector0 = new ErrorCollector();
      // Undeclared exception!
      try { 
        IRFactory.transformTree(astRoot0, "YA+>4Wc", (Config) null, errorCollector0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
      }
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      LinkedHashSet<String> linkedHashSet0 = new LinkedHashSet<String>();
      NumberLiteral numberLiteral0 = new NumberLiteral(0, "EXTERNS", 19);
      astRoot0.addChild(numberLiteral0);
      Config config0 = new Config(linkedHashSet0, linkedHashSet0, false, true, false);
      Node node0 = IRFactory.transformTree(astRoot0, "/J!h._7=", config0, (ErrorReporter) null);
      assertEquals(132, node0.getType());
      assertTrue(node0.hasOneChild());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      LinkedHashSet<String> linkedHashSet0 = new LinkedHashSet<String>();
      ThrowStatement throwStatement0 = new ThrowStatement(0);
      astRoot0.addChildToFront(throwStatement0);
      Config config0 = new Config(linkedHashSet0, linkedHashSet0, true, true, true);
      Context.getCurrentContext();
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      Charset charset0 = Charset.defaultCharset();
      Set<String> set0 = charset0.aliases();
      NewExpression newExpression0 = new NewExpression();
      astRoot0.addChild(newExpression0);
      Config config0 = new Config(set0, set0, true, true, true);
      // Undeclared exception!
      try { 
        IRFactory.transformTree(astRoot0, (String) null, config0, (ErrorReporter) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
      }
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      TreeSet<String> treeSet0 = new TreeSet<String>();
      StringLiteral stringLiteral0 = new StringLiteral(8);
      astRoot0.addChild(stringLiteral0);
      Config config0 = new Config(treeSet0, treeSet0, false, false, false);
      // Undeclared exception!
      try { 
        IRFactory.transformTree(astRoot0, "DY@vlY`5<~z", config0, (ErrorReporter) null);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // StringNode: str is null
         //
         verifyException("com.google.javascript.rhino.Node$StringNode", e);
      }
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      LinkedHashSet<String> linkedHashSet0 = new LinkedHashSet<String>();
      Scope scope0 = new Scope(105);
      astRoot0.addChild(scope0);
      ToolErrorReporter toolErrorReporter0 = new ToolErrorReporter(false);
      Config config0 = new Config(linkedHashSet0, linkedHashSet0, false, true, false);
      Node node0 = IRFactory.transformTree(astRoot0, "o1wl", config0, toolErrorReporter0);
      assertEquals(132, node0.getType());
      assertTrue(node0.hasOneChild());
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      LinkedHashSet<String> linkedHashSet0 = new LinkedHashSet<String>();
      ExpressionStatement expressionStatement0 = new ExpressionStatement(astRoot0);
      astRoot0.addChildToFront(expressionStatement0);
      Config config0 = new Config(linkedHashSet0, linkedHashSet0, false, false, false);
      // Undeclared exception!
      try { 
        IRFactory.transformTree(astRoot0, "const", config0, (ErrorReporter) null);
        fail("Expecting exception: StackOverflowError");
      
      } catch(StackOverflowError e) {
         //
         // no message in exception (getMessage() returned null)
         //
      }
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      LinkedHashSet<String> linkedHashSet0 = new LinkedHashSet<String>();
      WithStatement withStatement0 = new WithStatement(1, 1);
      astRoot0.addChild(withStatement0);
      Config config0 = new Config(linkedHashSet0, linkedHashSet0, false, false, false);
      // Undeclared exception!
      try { 
        IRFactory.transformTree(astRoot0, (String) null, config0, (ErrorReporter) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
      }
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      LinkedHashSet<String> linkedHashSet0 = new LinkedHashSet<String>();
      ArrayComprehensionLoop arrayComprehensionLoop0 = new ArrayComprehensionLoop();
      astRoot0.addChildToFront(arrayComprehensionLoop0);
      Config config0 = new Config(linkedHashSet0, linkedHashSet0, false, false, true);
      // Undeclared exception!
      try { 
        IRFactory.transformTree(astRoot0, "", config0, (ErrorReporter) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
      }
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      LinkedHashSet<String> linkedHashSet0 = new LinkedHashSet<String>();
      ForLoop forLoop0 = new ForLoop(1);
      astRoot0.addChild(forLoop0);
      Config config0 = new Config(linkedHashSet0, linkedHashSet0, true, true, true);
      // Undeclared exception!
      try { 
        IRFactory.transformTree(astRoot0, "", config0, (ErrorReporter) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
      }
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      LinkedHashSet<String> linkedHashSet0 = new LinkedHashSet<String>();
      ConditionalExpression conditionalExpression0 = new ConditionalExpression();
      astRoot0.addChild(conditionalExpression0);
      Config config0 = new Config(linkedHashSet0, linkedHashSet0, true, true, false);
      // Undeclared exception!
      try { 
        IRFactory.transformTree(astRoot0, "", config0, (ErrorReporter) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
      }
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      IfStatement ifStatement0 = new IfStatement(3);
      ifStatement0.setType(0);
      astRoot0.addChildToFront(ifStatement0);
      LinkedHashSet<String> linkedHashSet0 = new LinkedHashSet<String>();
      Config config0 = new Config(linkedHashSet0, linkedHashSet0, false, false, false);
      // Undeclared exception!
      try { 
        IRFactory.transformTree(astRoot0, "U9)1\u0004N>!Vt5]ywh9z", config0, (ErrorReporter) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.parsing.IRFactory$TransformDispatcher", e);
      }
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      LinkedHashSet<String> linkedHashSet0 = new LinkedHashSet<String>();
      Config config0 = new Config(linkedHashSet0, linkedHashSet0, true, true, true);
      Token.CommentType token_CommentType0 = Token.CommentType.JSDOC;
      Comment comment0 = new Comment(1753, 17, token_CommentType0, "U5+.^REoMY6sxdU84F@");
      astRoot0.addComment(comment0);
      // Undeclared exception!
      try { 
        IRFactory.transformTree(astRoot0, ".9gO%ARS", config0, (ErrorReporter) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.parsing.JsDocInfoParser$ErrorReporterParser", e);
      }
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      Token.CommentType token_CommentType0 = Token.CommentType.LINE;
      Comment comment0 = new Comment(0, 2, token_CommentType0, "Cy|#&i");
      astRoot0.addComment(comment0);
      ErrorCollector errorCollector0 = new ErrorCollector();
      Node node0 = IRFactory.transformTree(astRoot0, "r<xn^Q0", (Config) null, errorCollector0);
      assertEquals(132, node0.getType());
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      LinkedHashSet<String> linkedHashSet0 = new LinkedHashSet<String>();
      Token.CommentType token_CommentType0 = Token.CommentType.LINE;
      Comment comment0 = new Comment(8, 8, token_CommentType0, "java/lang/Baoolean");
      astRoot0.setJsDocNode(comment0);
      Config config0 = new Config(linkedHashSet0, linkedHashSet0, false, false, false);
      ToolErrorReporter toolErrorReporter0 = new ToolErrorReporter(false);
      Node node0 = IRFactory.transformTree(astRoot0, "java/lang/Baoolean", config0, toolErrorReporter0);
      assertEquals(132, node0.getType());
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot(31);
      LinkedHashSet<String> linkedHashSet0 = new LinkedHashSet<String>();
      Config config0 = new Config(linkedHashSet0, linkedHashSet0, false, true, true);
      Node node0 = IRFactory.transformTree(astRoot0, "\n\nSubtree1: ", config0, (ErrorReporter) null);
      assertEquals(132, node0.getType());
      assertEquals((-1), node0.getLineno());
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      LinkedHashSet<String> linkedHashSet0 = new LinkedHashSet<String>();
      Block block0 = new Block();
      block0.addChildToFront(astRoot0);
      astRoot0.addChild(block0);
      Config config0 = new Config(linkedHashSet0, linkedHashSet0, false, false, false);
      MockFile mockFile0 = new MockFile("Ii}ihULgX<l3WJ{J", "compile");
      MockPrintStream mockPrintStream0 = new MockPrintStream(mockFile0);
      ToolErrorReporter toolErrorReporter0 = new ToolErrorReporter(true, mockPrintStream0);
      // Undeclared exception!
      try { 
        IRFactory.transformTree(astRoot0, "cn<uXZ Tg`Z4HR4i(", config0, toolErrorReporter0);
        fail("Expecting exception: StackOverflowError");
      
      } catch(StackOverflowError e) {
         //
         // no message in exception (getMessage() returned null)
         //
      }
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      ArrayLiteral arrayLiteral0 = new ArrayLiteral(2, 135);
      astRoot0.addChildToFront(arrayLiteral0);
      Node node0 = IRFactory.transformTree(astRoot0, "U3K7 rdc1cv", (Config) null, (ErrorReporter) null);
      assertEquals(1, node0.getChildCount());
      assertEquals(132, node0.getType());
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      LinkedHashSet<String> linkedHashSet0 = new LinkedHashSet<String>();
      BreakStatement breakStatement0 = new BreakStatement(0, 18);
      astRoot0.addChildToFront(breakStatement0);
      Config config0 = new Config(linkedHashSet0, linkedHashSet0, true, true, true);
      ToolErrorReporter toolErrorReporter0 = new ToolErrorReporter(true);
      Node node0 = IRFactory.transformTree(astRoot0, "&!Ax8]h", config0, toolErrorReporter0);
      assertEquals(132, node0.getType());
      assertEquals(1, node0.getChildCount());
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      ContinueStatement continueStatement0 = new ContinueStatement(22, 2);
      astRoot0.addChild(continueStatement0);
      Node node0 = IRFactory.transformTree(astRoot0, "escape", (Config) null, (ErrorReporter) null);
      assertEquals(132, node0.getType());
      assertEquals(1, node0.getChildCount());
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      ObjectLiteral objectLiteral0 = new ObjectLiteral((-3200));
      astRoot0.addChildToFront(objectLiteral0);
      Node node0 = IRFactory.transformTree(astRoot0, "escape", (Config) null, (ErrorReporter) null);
      assertEquals(132, node0.getType());
      assertEquals(1, node0.getChildCount());
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      ReturnStatement returnStatement0 = new ReturnStatement();
      astRoot0.addChild(returnStatement0);
      ErrorCollector errorCollector0 = new ErrorCollector();
      Node node0 = IRFactory.transformTree(astRoot0, "r<xn^Q0", (Config) null, errorCollector0);
      assertEquals(132, node0.getType());
      assertTrue(node0.hasOneChild());
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      SwitchCase switchCase0 = new SwitchCase(8, 4);
      switchCase0.setExpression(astRoot0);
      astRoot0.addChild(switchCase0);
      Context context0 = new Context();
      ErrorReporter errorReporter0 = context0.getErrorReporter();
      // Undeclared exception!
      try { 
        IRFactory.transformTree(astRoot0, "language version", (Config) null, errorReporter0);
        fail("Expecting exception: StackOverflowError");
      
      } catch(StackOverflowError e) {
         //
         // no message in exception (getMessage() returned null)
         //
      }
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      SwitchCase switchCase0 = new SwitchCase(8, 4);
      astRoot0.addChild(switchCase0);
      Context context0 = new Context();
      ErrorReporter errorReporter0 = context0.getErrorReporter();
      Node node0 = IRFactory.transformTree(astRoot0, "language version", (Config) null, errorReporter0);
      assertEquals(132, node0.getType());
      assertTrue(node0.hasOneChild());
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      FunctionCall functionCall0 = new FunctionCall();
      astRoot0.addChild(functionCall0);
      ErrorCollector errorCollector0 = new ErrorCollector();
      // Undeclared exception!
      try { 
        IRFactory.transformTree(astRoot0, "setters are not supported in Internet Explorer", (Config) null, errorCollector0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
      }
  }
}