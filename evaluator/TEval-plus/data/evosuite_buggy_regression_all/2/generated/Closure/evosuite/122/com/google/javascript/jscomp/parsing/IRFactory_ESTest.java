/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 14:37:24 GMT 2023
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
import com.google.javascript.rhino.head.ast.AstRoot;
import com.google.javascript.rhino.head.ast.Block;
import com.google.javascript.rhino.head.ast.BreakStatement;
import com.google.javascript.rhino.head.ast.Comment;
import com.google.javascript.rhino.head.ast.ContinueStatement;
import com.google.javascript.rhino.head.ast.DoLoop;
import com.google.javascript.rhino.head.ast.ElementGet;
import com.google.javascript.rhino.head.ast.EmptyExpression;
import com.google.javascript.rhino.head.ast.EmptyStatement;
import com.google.javascript.rhino.head.ast.ErrorCollector;
import com.google.javascript.rhino.head.ast.ForInLoop;
import com.google.javascript.rhino.head.ast.ForLoop;
import com.google.javascript.rhino.head.ast.FunctionCall;
import com.google.javascript.rhino.head.ast.Label;
import com.google.javascript.rhino.head.ast.LabeledStatement;
import com.google.javascript.rhino.head.ast.Name;
import com.google.javascript.rhino.head.ast.NumberLiteral;
import com.google.javascript.rhino.head.ast.ObjectLiteral;
import com.google.javascript.rhino.head.ast.ObjectProperty;
import com.google.javascript.rhino.head.ast.ParenthesizedExpression;
import com.google.javascript.rhino.head.ast.ReturnStatement;
import com.google.javascript.rhino.head.ast.StringLiteral;
import com.google.javascript.rhino.head.ast.SwitchCase;
import com.google.javascript.rhino.head.ast.ThrowStatement;
import com.google.javascript.rhino.head.ast.VariableDeclaration;
import com.google.javascript.rhino.head.tools.ToolErrorReporter;
import com.google.javascript.rhino.jstype.SimpleSourceFile;
import com.google.javascript.rhino.jstype.StaticSourceFile;
import java.util.Locale;
import java.util.Set;
import java.util.TreeSet;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class IRFactory_ESTest extends IRFactory_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      SimpleSourceFile simpleSourceFile0 = new SimpleSourceFile("", false);
      ObjectProperty objectProperty0 = new ObjectProperty();
      ParenthesizedExpression parenthesizedExpression0 = new ParenthesizedExpression(objectProperty0);
      astRoot0.addChildToFront(parenthesizedExpression0);
      TreeSet<String> treeSet0 = new TreeSet<String>();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5;
      Config config0 = new Config(treeSet0, treeSet0, true, config_LanguageMode0, false);
      // Undeclared exception!
      try { 
        IRFactory.transformTree(astRoot0, simpleSourceFile0, "", config0, (ErrorReporter) null);
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // 103
         //
         verifyException("com.google.javascript.jscomp.parsing.IRFactory", e);
      }
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      DoLoop doLoop0 = new DoLoop(25);
      astRoot0.addChildToBack(doLoop0);
      TreeSet<String> treeSet0 = new TreeSet<String>();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5_STRICT;
      Config config0 = new Config(treeSet0, treeSet0, true, config_LanguageMode0, true);
      ErrorCollector errorCollector0 = new ErrorCollector();
      // Undeclared exception!
      try { 
        IRFactory.transformTree(astRoot0, (StaticSourceFile) null, "", config0, errorCollector0);
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
      SimpleSourceFile simpleSourceFile0 = new SimpleSourceFile("9O", true);
      EmptyExpression emptyExpression0 = new EmptyExpression(11, 2240);
      astRoot0.addChildToFront(emptyExpression0);
      TreeSet<String> treeSet0 = new TreeSet<String>();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5;
      Config config0 = new Config(treeSet0, treeSet0, true, config_LanguageMode0, false);
      Context context0 = Context.enter();
      ErrorReporter errorReporter0 = context0.getErrorReporter();
      Node node0 = IRFactory.transformTree(astRoot0, simpleSourceFile0, "9O", config0, errorReporter0);
      assertEquals(1, node0.getLength());
      assertTrue(node0.isScript());
      assertTrue(node0.isFromExterns());
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      SimpleSourceFile simpleSourceFile0 = new SimpleSourceFile("$;i3#\"h[S|(", true);
      ElementGet elementGet0 = new ElementGet(astRoot0, astRoot0);
      astRoot0.addChildToFront(elementGet0);
      TreeSet<String> treeSet0 = new TreeSet<String>();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5;
      Config config0 = new Config(treeSet0, treeSet0, true, config_LanguageMode0, true);
      // Undeclared exception!
      try { 
        IRFactory.transformTree(astRoot0, simpleSourceFile0, "$;i3#\"h[S|(", config0, (ErrorReporter) null);
        fail("Expecting exception: StackOverflowError");
      
      } catch(StackOverflowError e) {
         //
         // no message in exception (getMessage() returned null)
         //
      }
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      SimpleSourceFile simpleSourceFile0 = new SimpleSourceFile("O", false);
      ForLoop forLoop0 = new ForLoop();
      astRoot0.addChild(forLoop0);
      TreeSet<String> treeSet0 = new TreeSet<String>();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5;
      Config config0 = new Config(treeSet0, treeSet0, false, config_LanguageMode0, true);
      // Undeclared exception!
      try { 
        IRFactory.transformTree(astRoot0, simpleSourceFile0, "O", config0, (ErrorReporter) null);
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
      SwitchCase switchCase0 = new SwitchCase(1, 1);
      Label label0 = new Label(8, (-1136), "NHsE[Vuy=j");
      switchCase0.addStatement(label0);
      astRoot0.addChildToFront(switchCase0);
      TreeSet<String> treeSet0 = new TreeSet<String>();
      SimpleSourceFile simpleSourceFile0 = new SimpleSourceFile("com.google.javascript.jscomp.parsing.IRFactory$TransformDispatcher", false);
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5_STRICT;
      Config config0 = new Config(treeSet0, treeSet0, false, config_LanguageMode0, false);
      Node node0 = IRFactory.transformTree(astRoot0, simpleSourceFile0, "NHsE[Vuy=j", config0, (ErrorReporter) null);
      assertEquals(0, node0.getLength());
      assertEquals(132, node0.getType());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      SimpleSourceFile simpleSourceFile0 = new SimpleSourceFile("n", false);
      NumberLiteral numberLiteral0 = new NumberLiteral();
      astRoot0.addChildToFront(numberLiteral0);
      TreeSet<String> treeSet0 = new TreeSet<String>();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT3;
      Config config0 = new Config(treeSet0, treeSet0, false, config_LanguageMode0, true);
      ErrorCollector errorCollector0 = new ErrorCollector();
      Node node0 = IRFactory.transformTree(astRoot0, simpleSourceFile0, "n", config0, errorCollector0);
      assertEquals(0, node0.getLength());
      assertEquals(132, node0.getType());
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      EmptyStatement emptyStatement0 = new EmptyStatement((-149));
      astRoot0.addChildToBack(emptyStatement0);
      TreeSet<String> treeSet0 = new TreeSet<String>();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT3;
      Config config0 = new Config(treeSet0, treeSet0, true, config_LanguageMode0, false);
      Node node0 = IRFactory.transformTree(astRoot0, (StaticSourceFile) null, "9-x}Cwuc", config0, (ErrorReporter) null);
      assertEquals(1, node0.getLength());
      assertTrue(node0.isScript());
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      Block block0 = new Block((-1638));
      astRoot0.addChildrenToBack(block0);
      SimpleSourceFile simpleSourceFile0 = new SimpleSourceFile("D1", false);
      TreeSet<String> treeSet0 = new TreeSet<String>();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT3;
      Config config0 = new Config(treeSet0, treeSet0, false, config_LanguageMode0, false);
      Node node0 = IRFactory.transformTree(astRoot0, simpleSourceFile0, "D1", config0, (ErrorReporter) null);
      assertEquals(0, node0.getLength());
      assertEquals(132, node0.getType());
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      Name name0 = new Name(0, 23);
      astRoot0.addChildToFront(name0);
      TreeSet<String> treeSet0 = new TreeSet<String>();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5_STRICT;
      Config config0 = new Config(treeSet0, treeSet0, true, config_LanguageMode0, true);
      // Undeclared exception!
      try { 
        IRFactory.transformTree(astRoot0, (StaticSourceFile) null, "", config0, (ErrorReporter) null);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // StringNode: str is null
         //
         verifyException("com.google.javascript.rhino.Node$StringNode", e);
      }
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      SimpleSourceFile simpleSourceFile0 = new SimpleSourceFile("", true);
      Token.CommentType token_CommentType0 = Token.CommentType.BLOCK_COMMENT;
      Comment comment0 = new Comment(9, 1, token_CommentType0, "");
      astRoot0.addChildToFront(comment0);
      TreeSet<String> treeSet0 = new TreeSet<String>();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5_STRICT;
      Config config0 = new Config(treeSet0, treeSet0, true, config_LanguageMode0, true);
      // Undeclared exception!
      try { 
        IRFactory.transformTree(astRoot0, simpleSourceFile0, "", config0, (ErrorReporter) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.parsing.IRFactory$TransformDispatcher", e);
      }
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      SimpleSourceFile simpleSourceFile0 = new SimpleSourceFile("a&M@A=qcr(Ub&*", false);
      ThrowStatement throwStatement0 = new ThrowStatement(23);
      astRoot0.addChildToFront(throwStatement0);
      TreeSet<String> treeSet0 = new TreeSet<String>();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5_STRICT;
      Config config0 = new Config(treeSet0, treeSet0, false, config_LanguageMode0, false);
      // Undeclared exception!
      try { 
        IRFactory.transformTree(astRoot0, simpleSourceFile0, "", config0, (ErrorReporter) null);
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
      Token.CommentType token_CommentType0 = Token.CommentType.JSDOC;
      Comment comment0 = new Comment(6, 6, token_CommentType0, "getters are not supported in older versions of JavaScript. If you are targeting newer versions of JavaScript, set the appropriate language_in option.");
      astRoot0.addComment(comment0);
      Locale locale0 = Locale.FRENCH;
      Set<String> set0 = locale0.getUnicodeLocaleAttributes();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5;
      Config config0 = new Config(set0, set0, false, config_LanguageMode0, false);
      ToolErrorReporter toolErrorReporter0 = new ToolErrorReporter(false);
      SimpleSourceFile simpleSourceFile0 = new SimpleSourceFile("getters are not supported in older versions of JavaScript. If you are targeting newer versions of JavaScript, set the appropriate language_in option.", false);
      Node node0 = IRFactory.transformTree(astRoot0, simpleSourceFile0, "getters are not supported in older versions of JavaScript. If you are targeting newer versions of JavaScript, set the appropriate language_in option.", config0, toolErrorReporter0);
      assertTrue(node0.isScript());
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      SimpleSourceFile simpleSourceFile0 = new SimpleSourceFile("g .a", false);
      StringLiteral stringLiteral0 = new StringLiteral((-2385));
      astRoot0.addChildToFront(stringLiteral0);
      TreeSet<String> treeSet0 = new TreeSet<String>();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5;
      Config config0 = new Config(treeSet0, treeSet0, false, config_LanguageMode0, false);
      // Undeclared exception!
      try { 
        IRFactory.transformTree(astRoot0, simpleSourceFile0, "g .a", config0, (ErrorReporter) null);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // StringNode: str is null
         //
         verifyException("com.google.javascript.rhino.Node$StringNode", e);
      }
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      SimpleSourceFile simpleSourceFile0 = new SimpleSourceFile("", true);
      Token.CommentType token_CommentType0 = Token.CommentType.LINE;
      Comment comment0 = new Comment(9, 1, token_CommentType0, "");
      astRoot0.addComment(comment0);
      TreeSet<String> treeSet0 = new TreeSet<String>();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5_STRICT;
      Config config0 = new Config(treeSet0, treeSet0, true, config_LanguageMode0, true);
      Node node0 = IRFactory.transformTree(astRoot0, simpleSourceFile0, "", config0, (ErrorReporter) null);
      assertEquals(1, node0.getLength());
      assertTrue(node0.isScript());
      assertTrue(node0.isFromExterns());
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      SimpleSourceFile simpleSourceFile0 = new SimpleSourceFile("", true);
      Token.CommentType token_CommentType0 = Token.CommentType.BLOCK_COMMENT;
      Comment comment0 = new Comment(9, 1, token_CommentType0, "");
      astRoot0.addComment(comment0);
      TreeSet<String> treeSet0 = new TreeSet<String>();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT3;
      Config config0 = new Config(treeSet0, treeSet0, true, config_LanguageMode0, true);
      Node node0 = IRFactory.transformTree(astRoot0, simpleSourceFile0, "", config0, (ErrorReporter) null);
      assertTrue(node0.isScript());
      assertTrue(node0.isFromExterns());
      assertEquals(1, node0.getLength());
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot(119);
      Context context0 = Context.getCurrentContext();
      Locale locale0 = context0.getLocale();
      Set<String> set0 = locale0.getUnicodeLocaleAttributes();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5_STRICT;
      Config config0 = new Config(set0, set0, true, config_LanguageMode0, false);
      ErrorCollector errorCollector0 = new ErrorCollector();
      Node node0 = IRFactory.transformTree(astRoot0, (StaticSourceFile) null, "*/}\n", config0, errorCollector0);
      assertEquals(1, node0.getLength());
      assertEquals((-1), node0.getCharno());
      assertEquals(132, node0.getType());
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      SimpleSourceFile simpleSourceFile0 = new SimpleSourceFile("O", false);
      BreakStatement breakStatement0 = new BreakStatement();
      astRoot0.addChildToFront(breakStatement0);
      TreeSet<String> treeSet0 = new TreeSet<String>();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5;
      Config config0 = new Config(treeSet0, treeSet0, false, config_LanguageMode0, true);
      Node node0 = IRFactory.transformTree(astRoot0, simpleSourceFile0, "O", config0, (ErrorReporter) null);
      assertEquals(0, node0.getLength());
      assertEquals(132, node0.getType());
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      SimpleSourceFile simpleSourceFile0 = new SimpleSourceFile("397s.j1L9RNejf", false);
      ContinueStatement continueStatement0 = new ContinueStatement((Name) null);
      astRoot0.addChildToFront(continueStatement0);
      TreeSet<String> treeSet0 = new TreeSet<String>();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5_STRICT;
      Config config0 = new Config(treeSet0, treeSet0, false, config_LanguageMode0, false);
      Node node0 = IRFactory.transformTree(astRoot0, simpleSourceFile0, "397s.j1L9RNejf", config0, (ErrorReporter) null);
      assertEquals(0, node0.getLength());
      assertEquals(132, node0.getType());
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      SimpleSourceFile simpleSourceFile0 = new SimpleSourceFile("gZ&9H3%v*l^`", true);
      ForInLoop forInLoop0 = new ForInLoop(5, 16);
      astRoot0.addChildToFront(forInLoop0);
      TreeSet<String> treeSet0 = new TreeSet<String>();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT3;
      Config config0 = new Config(treeSet0, treeSet0, true, config_LanguageMode0, true);
      // Undeclared exception!
      try { 
        IRFactory.transformTree(astRoot0, simpleSourceFile0, "gZ&9H3%v*l^`", config0, (ErrorReporter) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
      }
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      SimpleSourceFile simpleSourceFile0 = new SimpleSourceFile("M?vvsz>", true);
      LabeledStatement labeledStatement0 = new LabeledStatement();
      astRoot0.addChildToFront(labeledStatement0);
      TreeSet<String> treeSet0 = new TreeSet<String>();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5;
      Config config0 = new Config(treeSet0, treeSet0, true, config_LanguageMode0, false);
      // Undeclared exception!
      try { 
        IRFactory.transformTree(astRoot0, simpleSourceFile0, "M?vvsz>", config0, (ErrorReporter) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
      }
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      Name name0 = new Name(0, 23);
      astRoot0.addChildToFront(name0);
      TreeSet<String> treeSet0 = new TreeSet<String>();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT3;
      Config config0 = new Config(treeSet0, treeSet0, true, config_LanguageMode0, true);
      SimpleSourceFile simpleSourceFile0 = new SimpleSourceFile("Keywords and reserved words are not allowed as unquoted property names in older versions of JavaScript. If you are targeting newer versions of JavaScript, set the appropriate language_in option.", true);
      ErrorCollector errorCollector0 = new ErrorCollector();
      // Undeclared exception!
      try { 
        IRFactory.transformTree(astRoot0, simpleSourceFile0, "", config0, errorCollector0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.rhino.TokenStream", e);
      }
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      SimpleSourceFile simpleSourceFile0 = new SimpleSourceFile("", true);
      ObjectLiteral objectLiteral0 = new ObjectLiteral(10);
      astRoot0.addChildToFront(objectLiteral0);
      TreeSet<String> treeSet0 = new TreeSet<String>();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5;
      Config config0 = new Config(treeSet0, treeSet0, true, config_LanguageMode0, true);
      Node node0 = IRFactory.transformTree(astRoot0, simpleSourceFile0, "", config0, (ErrorReporter) null);
      assertEquals(1, node0.getLength());
      assertEquals(132, node0.getType());
      assertTrue(node0.isFromExterns());
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      ReturnStatement returnStatement0 = new ReturnStatement(1, 1);
      astRoot0.addChildToBack(returnStatement0);
      TreeSet<String> treeSet0 = new TreeSet<String>();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5;
      Config config0 = new Config(treeSet0, treeSet0, false, config_LanguageMode0, false);
      Node node0 = IRFactory.transformTree(astRoot0, (StaticSourceFile) null, "9-x}Cwuc", config0, (ErrorReporter) null);
      assertEquals(0, node0.getLength());
      assertEquals(132, node0.getType());
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      ReturnStatement returnStatement0 = new ReturnStatement(1, 1);
      returnStatement0.setReturnValue(astRoot0);
      astRoot0.addChildToBack(returnStatement0);
      TreeSet<String> treeSet0 = new TreeSet<String>();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5;
      Config config0 = new Config(treeSet0, treeSet0, false, config_LanguageMode0, true);
      // Undeclared exception!
      try { 
        IRFactory.transformTree(astRoot0, (StaticSourceFile) null, "9-x}Cwuc", config0, (ErrorReporter) null);
        fail("Expecting exception: StackOverflowError");
      
      } catch(StackOverflowError e) {
         //
         // no message in exception (getMessage() returned null)
         //
      }
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      SimpleSourceFile simpleSourceFile0 = new SimpleSourceFile("gZ&9H3%v*l^`", false);
      SwitchCase switchCase0 = new SwitchCase(1, 1);
      astRoot0.addChildToFront(switchCase0);
      TreeSet<String> treeSet0 = new TreeSet<String>();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5;
      Config config0 = new Config(treeSet0, treeSet0, false, config_LanguageMode0, false);
      ErrorCollector errorCollector0 = new ErrorCollector();
      Node node0 = IRFactory.transformTree(astRoot0, simpleSourceFile0, "gZ&9H3%v*l^`", config0, errorCollector0);
      assertEquals(0, node0.getLength());
      assertEquals(132, node0.getType());
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      SimpleSourceFile simpleSourceFile0 = new SimpleSourceFile("397s.j1L9RNejf", true);
      VariableDeclaration variableDeclaration0 = new VariableDeclaration(8, 2);
      astRoot0.addChildToFront(variableDeclaration0);
      TreeSet<String> treeSet0 = new TreeSet<String>();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5;
      Config config0 = new Config(treeSet0, treeSet0, true, config_LanguageMode0, true);
      Node node0 = IRFactory.transformTree(astRoot0, simpleSourceFile0, "397s.j1L9RNejf", config0, (ErrorReporter) null);
      assertEquals(1, node0.getLength());
      assertEquals(132, node0.getType());
      assertTrue(node0.isFromExterns());
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      SimpleSourceFile simpleSourceFile0 = new SimpleSourceFile("397s.j1L9RNejf", false);
      VariableDeclaration variableDeclaration0 = new VariableDeclaration();
      astRoot0.addChildToFront(variableDeclaration0);
      TreeSet<String> treeSet0 = new TreeSet<String>();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5_STRICT;
      Config config0 = new Config(treeSet0, treeSet0, false, config_LanguageMode0, false);
      Node node0 = IRFactory.transformTree(astRoot0, simpleSourceFile0, "397s.j1L9RNejf", config0, (ErrorReporter) null);
      assertEquals(0, node0.getLength());
      assertTrue(node0.isScript());
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      SimpleSourceFile simpleSourceFile0 = new SimpleSourceFile("", false);
      Locale locale0 = Locale.FRANCE;
      FunctionCall functionCall0 = new FunctionCall();
      astRoot0.addChildToFront(functionCall0);
      Set<String> set0 = locale0.getUnicodeLocaleAttributes();
      TreeSet<String> treeSet0 = new TreeSet<String>();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5;
      Config config0 = new Config(treeSet0, set0, true, config_LanguageMode0, false);
      // Undeclared exception!
      try { 
        IRFactory.transformTree(astRoot0, simpleSourceFile0, "|X-=R,c|4+H4", config0, (ErrorReporter) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
      }
  }
}
