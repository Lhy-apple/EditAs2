/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 02:09:17 GMT 2023
 */

package com.google.javascript.jscomp.parsing;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.google.javascript.jscomp.parsing.Config;
import com.google.javascript.jscomp.parsing.IRFactory;
import com.google.javascript.rhino.Node;
import com.google.javascript.rhino.head.Context;
import com.google.javascript.rhino.head.ErrorReporter;
import com.google.javascript.rhino.head.Token;
import com.google.javascript.rhino.head.ast.ArrayComprehension;
import com.google.javascript.rhino.head.ast.ArrayComprehensionLoop;
import com.google.javascript.rhino.head.ast.ArrayLiteral;
import com.google.javascript.rhino.head.ast.AstRoot;
import com.google.javascript.rhino.head.ast.Comment;
import com.google.javascript.rhino.head.ast.ConditionalExpression;
import com.google.javascript.rhino.head.ast.ContinueStatement;
import com.google.javascript.rhino.head.ast.EmptyExpression;
import com.google.javascript.rhino.head.ast.ErrorCollector;
import com.google.javascript.rhino.head.ast.FunctionCall;
import com.google.javascript.rhino.head.ast.FunctionNode;
import com.google.javascript.rhino.head.ast.Name;
import com.google.javascript.rhino.head.ast.ObjectLiteral;
import com.google.javascript.rhino.head.ast.ObjectProperty;
import com.google.javascript.rhino.head.ast.ParenthesizedExpression;
import com.google.javascript.rhino.head.ast.RegExpLiteral;
import com.google.javascript.rhino.head.ast.ReturnStatement;
import com.google.javascript.rhino.head.ast.Scope;
import com.google.javascript.rhino.head.ast.ThrowStatement;
import com.google.javascript.rhino.head.ast.VariableDeclaration;
import com.google.javascript.rhino.head.ast.WhileLoop;
import com.google.javascript.rhino.head.ast.WithStatement;
import com.google.javascript.rhino.head.tools.ToolErrorReporter;
import com.google.javascript.rhino.jstype.SimpleSourceFile;
import com.google.javascript.rhino.jstype.StaticSourceFile;
import java.util.TreeSet;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class IRFactory_ESTest extends IRFactory_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot(65136);
      SimpleSourceFile simpleSourceFile0 = new SimpleSourceFile("4H|Fy", true);
      TreeSet<String> treeSet0 = new TreeSet<String>();
      ConditionalExpression conditionalExpression0 = new ConditionalExpression();
      astRoot0.addChildrenToFront(conditionalExpression0);
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT3;
      Config config0 = new Config(treeSet0, treeSet0, true, config_LanguageMode0, true);
      ErrorCollector errorCollector0 = new ErrorCollector();
      // Undeclared exception!
      try { 
        IRFactory.transformTree(astRoot0, simpleSourceFile0, "", config0, errorCollector0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
      }
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      ParenthesizedExpression parenthesizedExpression0 = new ParenthesizedExpression((-1));
      astRoot0.addChildrenToFront(parenthesizedExpression0);
      SimpleSourceFile simpleSourceFile0 = new SimpleSourceFile("Zgb87<1", false);
      TreeSet<String> treeSet0 = new TreeSet<String>();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5_STRICT;
      Config config0 = new Config(treeSet0, treeSet0, true, config_LanguageMode0, false);
      ErrorCollector errorCollector0 = new ErrorCollector();
      // Undeclared exception!
      try { 
        IRFactory.transformTree(astRoot0, simpleSourceFile0, "4H|F", config0, errorCollector0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
      }
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot(65130);
      EmptyExpression emptyExpression0 = new EmptyExpression(1970);
      astRoot0.addChildrenToFront(emptyExpression0);
      SimpleSourceFile simpleSourceFile0 = new SimpleSourceFile("U@$6fi@YGXPhl", true);
      TreeSet<String> treeSet0 = new TreeSet<String>();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5;
      Config config0 = new Config(treeSet0, treeSet0, true, config_LanguageMode0, true);
      ToolErrorReporter toolErrorReporter0 = new ToolErrorReporter(false);
      Node node0 = IRFactory.transformTree(astRoot0, simpleSourceFile0, "U@$6fi@YGXPhl", config0, toolErrorReporter0);
      assertEquals(1, node0.getLength());
      assertTrue(node0.isFromExterns());
      assertEquals((-1), node0.getLineno());
      assertEquals(132, node0.getType());
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      ObjectProperty objectProperty0 = new ObjectProperty(8);
      astRoot0.addChildrenToFront(objectProperty0);
      SimpleSourceFile simpleSourceFile0 = new SimpleSourceFile((String) null, true);
      TreeSet<String> treeSet0 = new TreeSet<String>();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5;
      Config config0 = new Config(treeSet0, treeSet0, false, config_LanguageMode0, true);
      ErrorCollector errorCollector0 = new ErrorCollector();
      // Undeclared exception!
      try { 
        IRFactory.transformTree(astRoot0, simpleSourceFile0, (String) null, config0, errorCollector0);
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // 103
         //
         verifyException("com.google.javascript.jscomp.parsing.IRFactory", e);
      }
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      WithStatement withStatement0 = new WithStatement(10);
      astRoot0.addChildToFront(withStatement0);
      SimpleSourceFile simpleSourceFile0 = new SimpleSourceFile((String) null, true);
      TreeSet<String> treeSet0 = new TreeSet<String>();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT3;
      Config config0 = new Config(treeSet0, treeSet0, true, config_LanguageMode0, true);
      ErrorCollector errorCollector0 = new ErrorCollector();
      // Undeclared exception!
      try { 
        IRFactory.transformTree(astRoot0, simpleSourceFile0, (String) null, config0, errorCollector0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
      }
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot(65130);
      com.google.javascript.rhino.head.Node node0 = com.google.javascript.rhino.head.Node.newNumber((-2708.965967337034));
      astRoot0.addChildrenToFront(node0);
      SimpleSourceFile simpleSourceFile0 = new SimpleSourceFile("VM", false);
      TreeSet<String> treeSet0 = new TreeSet<String>();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5;
      Config config0 = new Config(treeSet0, treeSet0, false, config_LanguageMode0, true);
      ErrorCollector errorCollector0 = new ErrorCollector();
      Node node1 = IRFactory.transformTree(astRoot0, simpleSourceFile0, "~~S2P-n2`*4", config0, errorCollector0);
      assertEquals(0, node1.getLength());
      assertEquals((-1), node1.getCharno());
      assertEquals(1, node1.getChildCount());
      assertEquals(132, node1.getType());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      Scope scope0 = new Scope(11);
      astRoot0.addChildrenToFront(scope0);
      SimpleSourceFile simpleSourceFile0 = new SimpleSourceFile("4H|Fy", true);
      TreeSet<String> treeSet0 = new TreeSet<String>();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5_STRICT;
      Config config0 = new Config(treeSet0, treeSet0, true, config_LanguageMode0, true);
      ErrorCollector errorCollector0 = new ErrorCollector();
      Node node0 = IRFactory.transformTree(astRoot0, simpleSourceFile0, "4H|Fy", config0, errorCollector0);
      assertEquals(1, node0.getLength());
      assertTrue(node0.isFromExterns());
      assertTrue(node0.isScript());
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      WhileLoop whileLoop0 = new WhileLoop(2);
      astRoot0.addChildrenToFront(whileLoop0);
      SimpleSourceFile simpleSourceFile0 = new SimpleSourceFile("4H|Fy", true);
      TreeSet<String> treeSet0 = new TreeSet<String>();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5_STRICT;
      Config config0 = new Config(treeSet0, treeSet0, true, config_LanguageMode0, true);
      ErrorCollector errorCollector0 = new ErrorCollector();
      // Undeclared exception!
      try { 
        IRFactory.transformTree(astRoot0, simpleSourceFile0, "4H|Fy", config0, errorCollector0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
      }
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot(65130);
      Name name0 = new Name(2, "v?J,2h:~");
      FunctionNode functionNode0 = new FunctionNode((-1094), name0);
      astRoot0.addChildrenToFront(functionNode0);
      TreeSet<String> treeSet0 = new TreeSet<String>();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5_STRICT;
      Config config0 = new Config(treeSet0, treeSet0, true, config_LanguageMode0, true);
      ErrorCollector errorCollector0 = new ErrorCollector();
      SimpleSourceFile simpleSourceFile0 = new SimpleSourceFile("v?J,2h:~", true);
      // Undeclared exception!
      try { 
        IRFactory.transformTree(astRoot0, simpleSourceFile0, "v?J,2h:~", config0, errorCollector0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
      }
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      ArrayComprehension arrayComprehension0 = new ArrayComprehension(15, 0);
      astRoot0.addChildrenToFront(arrayComprehension0);
      SimpleSourceFile simpleSourceFile0 = new SimpleSourceFile("4H|Fy", true);
      TreeSet<String> treeSet0 = new TreeSet<String>();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5_STRICT;
      Config config0 = new Config(treeSet0, treeSet0, true, config_LanguageMode0, true);
      ErrorCollector errorCollector0 = new ErrorCollector();
      // Undeclared exception!
      try { 
        IRFactory.transformTree(astRoot0, simpleSourceFile0, "4H|Fy", config0, errorCollector0);
        fail("Expecting exception: UnsupportedOperationException");
      
      } catch(UnsupportedOperationException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.rhino.head.ast.ErrorCollector", e);
      }
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      ThrowStatement throwStatement0 = new ThrowStatement(4, 2);
      astRoot0.addChildrenToFront(throwStatement0);
      SimpleSourceFile simpleSourceFile0 = new SimpleSourceFile("__default_namespace__", true);
      TreeSet<String> treeSet0 = new TreeSet<String>();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5;
      Config config0 = new Config(treeSet0, treeSet0, false, config_LanguageMode0, false);
      ErrorCollector errorCollector0 = new ErrorCollector();
      // Undeclared exception!
      try { 
        IRFactory.transformTree(astRoot0, simpleSourceFile0, " GMT+", config0, errorCollector0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
      }
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      FunctionNode functionNode0 = new FunctionNode();
      AstRoot astRoot0 = new AstRoot(1);
      astRoot0.addChildrenToFront(functionNode0);
      SimpleSourceFile simpleSourceFile0 = new SimpleSourceFile("U@$6fi@YGXPhl", true);
      TreeSet<String> treeSet0 = new TreeSet<String>();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5;
      Config config0 = new Config(treeSet0, treeSet0, true, config_LanguageMode0, true);
      ToolErrorReporter toolErrorReporter0 = new ToolErrorReporter(true);
      Node node0 = IRFactory.transformTree(astRoot0, simpleSourceFile0, "U@$6fi@YGXPhl", config0, toolErrorReporter0);
      assertEquals(1, node0.getLength());
      assertTrue(node0.isFromExterns());
      assertEquals(Integer.MAX_VALUE, node0.getSourceOffset());
      assertTrue(node0.isScript());
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot(65130);
      RegExpLiteral regExpLiteral0 = new RegExpLiteral(8, 4);
      astRoot0.addChildrenToFront(regExpLiteral0);
      SimpleSourceFile simpleSourceFile0 = new SimpleSourceFile("D=J,2h~h", true);
      TreeSet<String> treeSet0 = new TreeSet<String>();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT3;
      Config config0 = new Config(treeSet0, treeSet0, true, config_LanguageMode0, true);
      ErrorCollector errorCollector0 = new ErrorCollector();
      // Undeclared exception!
      try { 
        IRFactory.transformTree(astRoot0, simpleSourceFile0, "D=J,2h~h", config0, errorCollector0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // StringNode: str is null
         //
         verifyException("com.google.javascript.rhino.Node$StringNode", e);
      }
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      SimpleSourceFile simpleSourceFile0 = new SimpleSourceFile("I `", true);
      TreeSet<String> treeSet0 = new TreeSet<String>();
      ErrorCollector errorCollector0 = new ErrorCollector();
      Token.CommentType token_CommentType0 = Token.CommentType.BLOCK_COMMENT;
      Comment comment0 = new Comment(1, 14, token_CommentType0, "?8^#)fR<>3YukOa^");
      astRoot0.addComment(comment0);
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5;
      Config config0 = new Config(treeSet0, treeSet0, true, config_LanguageMode0, true);
      Node node0 = IRFactory.transformTree(astRoot0, simpleSourceFile0, "?8^#)fR<>3YukOa^", config0, errorCollector0);
      assertTrue(node0.isFromExterns());
      assertEquals(1, node0.getLength());
      assertTrue(node0.isScript());
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot(1111);
      SimpleSourceFile simpleSourceFile0 = new SimpleSourceFile("bkg", true);
      TreeSet<String> treeSet0 = new TreeSet<String>();
      Token.CommentType token_CommentType0 = Token.CommentType.HTML;
      Comment comment0 = new Comment((-726), 8, token_CommentType0, "msg.no.semi.for");
      astRoot0.addComment(comment0);
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT3;
      Config config0 = new Config(treeSet0, treeSet0, true, config_LanguageMode0, false);
      ErrorCollector errorCollector0 = new ErrorCollector();
      Node node0 = IRFactory.transformTree(astRoot0, simpleSourceFile0, "bkg", config0, errorCollector0);
      assertTrue(node0.isFromExterns());
      assertTrue(node0.isScript());
      assertEquals((-1), node0.getLineno());
      assertEquals(1, node0.getLength());
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot(65136);
      SimpleSourceFile simpleSourceFile0 = new SimpleSourceFile("4H|Fy", true);
      TreeSet<String> treeSet0 = new TreeSet<String>();
      Token.CommentType token_CommentType0 = Token.CommentType.JSDOC;
      Comment comment0 = new Comment(13, 4, token_CommentType0, "4H|Fy");
      astRoot0.addComment(comment0);
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT3;
      Config config0 = new Config(treeSet0, treeSet0, true, config_LanguageMode0, true);
      Context context0 = new Context();
      ErrorReporter errorReporter0 = context0.getErrorReporter();
      Node node0 = IRFactory.transformTree(astRoot0, simpleSourceFile0, "error reporter", config0, errorReporter0);
      assertEquals(1, node0.getLength());
      assertEquals(132, node0.getType());
      assertTrue(node0.isFromExterns());
      assertEquals(Integer.MAX_VALUE, node0.getSourceOffset());
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot(0);
      SimpleSourceFile simpleSourceFile0 = new SimpleSourceFile("invalid assignment target", true);
      TreeSet<String> treeSet0 = new TreeSet<String>();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5_STRICT;
      Config config0 = new Config(treeSet0, treeSet0, true, config_LanguageMode0, false);
      Node node0 = IRFactory.transformTree(astRoot0, simpleSourceFile0, "\n * @", config0, (ErrorReporter) null);
      assertEquals(1, node0.getLength());
      assertTrue(node0.isScript());
      assertTrue(node0.isFromExterns());
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      ObjectLiteral objectLiteral0 = new ObjectLiteral(1, 532);
      astRoot0.addChildrenToFront(objectLiteral0);
      ObjectProperty objectProperty0 = new ObjectProperty(8);
      objectLiteral0.addElement(objectProperty0);
      SimpleSourceFile simpleSourceFile0 = new SimpleSourceFile("", false);
      TreeSet<String> treeSet0 = new TreeSet<String>();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT3;
      Config config0 = new Config(treeSet0, treeSet0, false, config_LanguageMode0, false);
      ErrorCollector errorCollector0 = new ErrorCollector();
      // Undeclared exception!
      try { 
        IRFactory.transformTree(astRoot0, simpleSourceFile0, "", config0, errorCollector0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
      }
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot(65130);
      ArrayLiteral arrayLiteral0 = new ArrayLiteral();
      astRoot0.addChildrenToFront(arrayLiteral0);
      TreeSet<String> treeSet0 = new TreeSet<String>();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5_STRICT;
      Config config0 = new Config(treeSet0, treeSet0, true, config_LanguageMode0, true);
      ErrorCollector errorCollector0 = new ErrorCollector();
      Node node0 = IRFactory.transformTree(astRoot0, (StaticSourceFile) null, "", config0, errorCollector0);
      assertEquals(1, node0.getLength());
      assertTrue(node0.isScript());
      assertEquals((-1), node0.getLineno());
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot(65136);
      ContinueStatement continueStatement0 = new ContinueStatement(16, 18, (Name) null);
      astRoot0.addChildrenToFront(continueStatement0);
      SimpleSourceFile simpleSourceFile0 = new SimpleSourceFile("", true);
      TreeSet<String> treeSet0 = new TreeSet<String>();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5_STRICT;
      Config config0 = new Config(treeSet0, treeSet0, true, config_LanguageMode0, true);
      ErrorCollector errorCollector0 = new ErrorCollector();
      Node node0 = IRFactory.transformTree(astRoot0, simpleSourceFile0, "", config0, errorCollector0);
      assertTrue(node0.isFromExterns());
      assertEquals((-1), node0.getLineno());
      assertEquals(1, node0.getLength());
      assertTrue(node0.isScript());
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      Name name0 = new Name(1, 8);
      ContinueStatement continueStatement0 = new ContinueStatement(name0);
      astRoot0.addChildrenToFront(continueStatement0);
      SimpleSourceFile simpleSourceFile0 = new SimpleSourceFile("U@$6fi@YGXPhl", false);
      TreeSet<String> treeSet0 = new TreeSet<String>();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5;
      Config config0 = new Config(treeSet0, treeSet0, false, config_LanguageMode0, true);
      ErrorCollector errorCollector0 = new ErrorCollector();
      // Undeclared exception!
      try { 
        IRFactory.transformTree(astRoot0, simpleSourceFile0, "U@$6fi@YGXPhl", config0, errorCollector0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // StringNode: str is null
         //
         verifyException("com.google.javascript.rhino.Node$StringNode", e);
      }
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot(65130);
      ArrayComprehensionLoop arrayComprehensionLoop0 = new ArrayComprehensionLoop();
      astRoot0.addChildrenToFront(arrayComprehensionLoop0);
      SimpleSourceFile simpleSourceFile0 = new SimpleSourceFile("v?J,2h:~", false);
      TreeSet<String> treeSet0 = new TreeSet<String>();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5;
      Config config0 = new Config(treeSet0, treeSet0, false, config_LanguageMode0, false);
      ErrorCollector errorCollector0 = new ErrorCollector();
      // Undeclared exception!
      try { 
        IRFactory.transformTree(astRoot0, simpleSourceFile0, "v?J,2h:~", config0, errorCollector0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
      }
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      Name name0 = new Name(16);
      FunctionNode functionNode0 = new FunctionNode(12, name0);
      astRoot0.addChildrenToFront(functionNode0);
      SimpleSourceFile simpleSourceFile0 = new SimpleSourceFile("-KGe??Bpy+U \u0004%", false);
      TreeSet<String> treeSet0 = new TreeSet<String>();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT3;
      Config config0 = new Config(treeSet0, treeSet0, false, config_LanguageMode0, false);
      ErrorCollector errorCollector0 = new ErrorCollector();
      // Undeclared exception!
      try { 
        IRFactory.transformTree(astRoot0, simpleSourceFile0, "-KGe??Bpy+U \u0004%", config0, errorCollector0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // StringNode: str is null
         //
         verifyException("com.google.javascript.rhino.Node$StringNode", e);
      }
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      ObjectLiteral objectLiteral0 = new ObjectLiteral(0, 0);
      astRoot0.addChildrenToFront(objectLiteral0);
      SimpleSourceFile simpleSourceFile0 = new SimpleSourceFile("", false);
      TreeSet<String> treeSet0 = new TreeSet<String>();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5;
      Config config0 = new Config(treeSet0, treeSet0, false, config_LanguageMode0, false);
      ErrorCollector errorCollector0 = new ErrorCollector();
      Node node0 = IRFactory.transformTree(astRoot0, simpleSourceFile0, "", config0, errorCollector0);
      assertEquals(0, node0.getLength());
      assertEquals(132, node0.getType());
      assertTrue(node0.hasChildren());
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      ObjectLiteral objectLiteral0 = new ObjectLiteral(1, 532);
      astRoot0.addChildrenToFront(objectLiteral0);
      ObjectProperty objectProperty0 = new ObjectProperty(8);
      objectLiteral0.addElement(objectProperty0);
      SimpleSourceFile simpleSourceFile0 = new SimpleSourceFile("", false);
      TreeSet<String> treeSet0 = new TreeSet<String>();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5_STRICT;
      Config config0 = new Config(treeSet0, treeSet0, false, config_LanguageMode0, false);
      ErrorCollector errorCollector0 = new ErrorCollector();
      // Undeclared exception!
      try { 
        IRFactory.transformTree(astRoot0, simpleSourceFile0, "", config0, errorCollector0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
      }
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot(1111);
      ReturnStatement returnStatement0 = new ReturnStatement(14, 2);
      astRoot0.addChildrenToFront(returnStatement0);
      SimpleSourceFile simpleSourceFile0 = new SimpleSourceFile("bkg", true);
      TreeSet<String> treeSet0 = new TreeSet<String>();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT3;
      Config config0 = new Config(treeSet0, treeSet0, true, config_LanguageMode0, false);
      ErrorCollector errorCollector0 = new ErrorCollector();
      Node node0 = IRFactory.transformTree(astRoot0, simpleSourceFile0, "bkg", config0, errorCollector0);
      assertEquals(1, node0.getLength());
      assertTrue(node0.isFromExterns());
      assertEquals(Integer.MAX_VALUE, node0.getSourceOffset());
      assertEquals(132, node0.getType());
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot(65130);
      ReturnStatement returnStatement0 = new ReturnStatement(8203, 134, astRoot0);
      astRoot0.addChildrenToFront(returnStatement0);
      SimpleSourceFile simpleSourceFile0 = new SimpleSourceFile("bkg", true);
      TreeSet<String> treeSet0 = new TreeSet<String>();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT3;
      Config config0 = new Config(treeSet0, treeSet0, false, config_LanguageMode0, true);
      ErrorCollector errorCollector0 = new ErrorCollector();
      // Undeclared exception!
      try { 
        IRFactory.transformTree(astRoot0, simpleSourceFile0, "bkg", config0, errorCollector0);
        fail("Expecting exception: StackOverflowError");
      
      } catch(StackOverflowError e) {
      }
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      VariableDeclaration variableDeclaration0 = new VariableDeclaration();
      astRoot0.addChildrenToFront(variableDeclaration0);
      SimpleSourceFile simpleSourceFile0 = new SimpleSourceFile("VM", false);
      TreeSet<String> treeSet0 = new TreeSet<String>();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5_STRICT;
      Config config0 = new Config(treeSet0, treeSet0, false, config_LanguageMode0, true);
      ErrorCollector errorCollector0 = new ErrorCollector();
      Node node0 = IRFactory.transformTree(astRoot0, simpleSourceFile0, "VM", config0, errorCollector0);
      assertEquals(0, node0.getLength());
      assertTrue(node0.hasOneChild());
      assertEquals(132, node0.getType());
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      VariableDeclaration variableDeclaration0 = new VariableDeclaration();
      astRoot0.addChildrenToFront(variableDeclaration0);
      SimpleSourceFile simpleSourceFile0 = new SimpleSourceFile("VM", false);
      TreeSet<String> treeSet0 = new TreeSet<String>();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5;
      Config config0 = new Config(treeSet0, treeSet0, false, config_LanguageMode0, false);
      ErrorCollector errorCollector0 = new ErrorCollector();
      Node node0 = IRFactory.transformTree(astRoot0, simpleSourceFile0, "VM", config0, errorCollector0);
      assertEquals(132, node0.getType());
      assertTrue(node0.hasOneChild());
      assertEquals(0, node0.getLength());
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot((-376));
      FunctionCall functionCall0 = new FunctionCall();
      astRoot0.addChildrenToFront(functionCall0);
      SimpleSourceFile simpleSourceFile0 = new SimpleSourceFile("create", false);
      TreeSet<String> treeSet0 = new TreeSet<String>();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT3;
      Config config0 = new Config(treeSet0, treeSet0, true, config_LanguageMode0, false);
      ErrorCollector errorCollector0 = new ErrorCollector();
      // Undeclared exception!
      try { 
        IRFactory.transformTree(astRoot0, simpleSourceFile0, "v?J,2h:~", config0, errorCollector0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
      }
  }
}
