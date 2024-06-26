/*
 * This file was automatically generated by EvoSuite
 * Sat Jul 29 18:01:29 GMT 2023
 */

package com.google.javascript.jscomp.parsing;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.google.javascript.jscomp.parsing.Config;
import com.google.javascript.jscomp.parsing.IRFactory;
import com.google.javascript.rhino.Node;
import com.google.javascript.rhino.head.ErrorReporter;
import com.google.javascript.rhino.head.Token;
import com.google.javascript.rhino.head.ast.ArrayComprehensionLoop;
import com.google.javascript.rhino.head.ast.ArrayLiteral;
import com.google.javascript.rhino.head.ast.Assignment;
import com.google.javascript.rhino.head.ast.AstRoot;
import com.google.javascript.rhino.head.ast.Block;
import com.google.javascript.rhino.head.ast.BreakStatement;
import com.google.javascript.rhino.head.ast.Comment;
import com.google.javascript.rhino.head.ast.ConditionalExpression;
import com.google.javascript.rhino.head.ast.ContinueStatement;
import com.google.javascript.rhino.head.ast.DoLoop;
import com.google.javascript.rhino.head.ast.ElementGet;
import com.google.javascript.rhino.head.ast.EmptyExpression;
import com.google.javascript.rhino.head.ast.ErrorCollector;
import com.google.javascript.rhino.head.ast.ExpressionStatement;
import com.google.javascript.rhino.head.ast.ForLoop;
import com.google.javascript.rhino.head.ast.FunctionCall;
import com.google.javascript.rhino.head.ast.FunctionNode;
import com.google.javascript.rhino.head.ast.Label;
import com.google.javascript.rhino.head.ast.LabeledStatement;
import com.google.javascript.rhino.head.ast.Name;
import com.google.javascript.rhino.head.ast.NewExpression;
import com.google.javascript.rhino.head.ast.NumberLiteral;
import com.google.javascript.rhino.head.ast.ObjectLiteral;
import com.google.javascript.rhino.head.ast.ObjectProperty;
import com.google.javascript.rhino.head.ast.ParenthesizedExpression;
import com.google.javascript.rhino.head.ast.PropertyGet;
import com.google.javascript.rhino.head.ast.RegExpLiteral;
import com.google.javascript.rhino.head.ast.ReturnStatement;
import com.google.javascript.rhino.head.ast.Scope;
import com.google.javascript.rhino.head.ast.SwitchCase;
import com.google.javascript.rhino.head.ast.ThrowStatement;
import com.google.javascript.rhino.head.ast.VariableDeclaration;
import com.google.javascript.rhino.head.ast.WhileLoop;
import com.google.javascript.rhino.head.ast.WithStatement;
import com.google.javascript.rhino.head.tools.ToolErrorReporter;
import com.google.javascript.rhino.jstype.SimpleSourceFile;
import com.google.javascript.rhino.jstype.StaticSourceFile;
import java.util.LinkedHashSet;
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
      TreeSet<String> treeSet0 = new TreeSet<String>();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5_STRICT;
      Config config0 = new Config(treeSet0, treeSet0, true, config_LanguageMode0, true);
      ConditionalExpression conditionalExpression0 = new ConditionalExpression(8, 4);
      astRoot0.addChildrenToFront(conditionalExpression0);
      ErrorCollector errorCollector0 = new ErrorCollector();
      // Undeclared exception!
      try { 
        IRFactory.transformTree(astRoot0, (StaticSourceFile) null, "f:fVId", config0, errorCollector0);
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
      TreeSet<String> treeSet0 = new TreeSet<String>();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5_STRICT;
      Config config0 = new Config(treeSet0, treeSet0, true, config_LanguageMode0, true);
      ParenthesizedExpression parenthesizedExpression0 = new ParenthesizedExpression(2, 16);
      astRoot0.addChildrenToFront(parenthesizedExpression0);
      ErrorCollector errorCollector0 = new ErrorCollector();
      // Undeclared exception!
      try { 
        IRFactory.transformTree(astRoot0, (StaticSourceFile) null, "\"p<:ZF|er", config0, errorCollector0);
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
      TreeSet<String> treeSet0 = new TreeSet<String>();
      DoLoop doLoop0 = new DoLoop();
      astRoot0.addChildrenToFront(doLoop0);
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT3;
      Config config0 = new Config(treeSet0, treeSet0, false, config_LanguageMode0, false);
      ErrorCollector errorCollector0 = new ErrorCollector();
      // Undeclared exception!
      try { 
        IRFactory.transformTree(astRoot0, (StaticSourceFile) null, "$0", config0, errorCollector0);
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
      TreeSet<String> treeSet0 = new TreeSet<String>();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5_STRICT;
      Config config0 = new Config(treeSet0, treeSet0, true, config_LanguageMode0, true);
      PropertyGet propertyGet0 = new PropertyGet();
      astRoot0.addChildrenToFront(propertyGet0);
      ErrorCollector errorCollector0 = new ErrorCollector();
      SimpleSourceFile simpleSourceFile0 = new SimpleSourceFile("interface", true);
      // Undeclared exception!
      try { 
        IRFactory.transformTree(astRoot0, simpleSourceFile0, "interface", config0, errorCollector0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
      }
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      Locale locale0 = Locale.KOREA;
      Set<String> set0 = locale0.getUnicodeLocaleKeys();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5_STRICT;
      Config config0 = new Config(set0, set0, false, config_LanguageMode0, false);
      EmptyExpression emptyExpression0 = new EmptyExpression();
      astRoot0.addChildrenToFront(emptyExpression0);
      ErrorCollector errorCollector0 = new ErrorCollector();
      Node node0 = IRFactory.transformTree(astRoot0, (StaticSourceFile) null, "/Kmafo/", config0, errorCollector0);
      assertEquals(0, node0.getLength());
      assertEquals(132, node0.getType());
      assertTrue(node0.hasOneChild());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      TreeSet<String> treeSet0 = new TreeSet<String>();
      ArrayComprehensionLoop arrayComprehensionLoop0 = new ArrayComprehensionLoop();
      astRoot0.addChildrenToFront(arrayComprehensionLoop0);
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5;
      Config config0 = new Config(treeSet0, treeSet0, false, config_LanguageMode0, false);
      ErrorCollector errorCollector0 = new ErrorCollector();
      // Undeclared exception!
      try { 
        IRFactory.transformTree(astRoot0, (StaticSourceFile) null, "Non-JSDoc comment has annotations. Did you mean to start it with '/**'?", config0, errorCollector0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
      }
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      TreeSet<String> treeSet0 = new TreeSet<String>();
      ElementGet elementGet0 = new ElementGet(0, 12);
      astRoot0.addChildrenToFront(elementGet0);
      SimpleSourceFile simpleSourceFile0 = new SimpleSourceFile("u", true);
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5_STRICT;
      Config config0 = new Config(treeSet0, treeSet0, true, config_LanguageMode0, true);
      ToolErrorReporter toolErrorReporter0 = new ToolErrorReporter(false);
      // Undeclared exception!
      try { 
        IRFactory.transformTree(astRoot0, simpleSourceFile0, "u", config0, toolErrorReporter0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
      }
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      TreeSet<String> treeSet0 = new TreeSet<String>();
      ObjectProperty objectProperty0 = new ObjectProperty();
      astRoot0.addChildrenToFront(objectProperty0);
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT3;
      Config config0 = new Config(treeSet0, treeSet0, true, config_LanguageMode0, true);
      ErrorCollector errorCollector0 = new ErrorCollector();
      // Undeclared exception!
      try { 
        IRFactory.transformTree(astRoot0, (StaticSourceFile) null, "$0", config0, errorCollector0);
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // 103
         //
         verifyException("com.google.javascript.jscomp.parsing.IRFactory", e);
      }
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      TreeSet<String> treeSet0 = new TreeSet<String>();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5_STRICT;
      WithStatement withStatement0 = new WithStatement();
      astRoot0.addChildrenToFront(withStatement0);
      Config config0 = new Config(treeSet0, treeSet0, true, config_LanguageMode0, false);
      ErrorCollector errorCollector0 = new ErrorCollector();
      // Undeclared exception!
      try { 
        IRFactory.transformTree(astRoot0, (StaticSourceFile) null, "$0", config0, errorCollector0);
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
      TreeSet<String> treeSet0 = new TreeSet<String>();
      ForLoop forLoop0 = new ForLoop(7);
      astRoot0.addChildrenToFront(forLoop0);
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT3;
      Config config0 = new Config(treeSet0, treeSet0, true, config_LanguageMode0, false);
      ErrorCollector errorCollector0 = new ErrorCollector();
      SimpleSourceFile simpleSourceFile0 = new SimpleSourceFile("Q\"mYcu/-&9'i/'%3=v", false);
      // Undeclared exception!
      try { 
        IRFactory.transformTree(astRoot0, simpleSourceFile0, "Non-JSDoc comment has annotations. Did you mean to start it with '/**'?", config0, errorCollector0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
      }
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      TreeSet<String> treeSet0 = new TreeSet<String>();
      Label label0 = new Label(1, 2);
      astRoot0.addChildrenToFront(label0);
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT3;
      Config config0 = new Config(treeSet0, treeSet0, true, config_LanguageMode0, true);
      ErrorCollector errorCollector0 = new ErrorCollector();
      // Undeclared exception!
      try { 
        IRFactory.transformTree(astRoot0, (StaticSourceFile) null, "$0", config0, errorCollector0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // StringNode: str is null
         //
         verifyException("com.google.javascript.rhino.Node$StringNode", e);
      }
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      Locale locale0 = Locale.KOREA;
      Set<String> set0 = locale0.getUnicodeLocaleKeys();
      NewExpression newExpression0 = new NewExpression(147);
      astRoot0.addChildrenToFront(newExpression0);
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT3;
      Config config0 = new Config(set0, set0, false, config_LanguageMode0, false);
      ErrorCollector errorCollector0 = new ErrorCollector();
      // Undeclared exception!
      try { 
        IRFactory.transformTree(astRoot0, (StaticSourceFile) null, "$0", config0, errorCollector0);
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
      TreeSet<String> treeSet0 = new TreeSet<String>();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5_STRICT;
      Config config0 = new Config(treeSet0, treeSet0, true, config_LanguageMode0, true);
      NumberLiteral numberLiteral0 = new NumberLiteral(12, 159);
      astRoot0.addChildrenToFront(numberLiteral0);
      ErrorCollector errorCollector0 = new ErrorCollector();
      Node node0 = IRFactory.transformTree(astRoot0, (StaticSourceFile) null, "w!Q5r^5", config0, errorCollector0);
      assertTrue(node0.isScript());
      assertEquals(1, node0.getLength());
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      Locale locale0 = Locale.KOREA;
      Set<String> set0 = locale0.getUnicodeLocaleAttributes();
      ExpressionStatement expressionStatement0 = new ExpressionStatement(1, 24);
      astRoot0.addChildrenToFront(expressionStatement0);
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5_STRICT;
      Config config0 = new Config(set0, set0, false, config_LanguageMode0, false);
      ErrorCollector errorCollector0 = new ErrorCollector();
      // Undeclared exception!
      try { 
        IRFactory.transformTree(astRoot0, (StaticSourceFile) null, "Non-JSDoc comment has annotations. Did you mean to start it with '/**'?", config0, errorCollector0);
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
      TreeSet<String> treeSet0 = new TreeSet<String>();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5_STRICT;
      Config config0 = new Config(treeSet0, treeSet0, false, config_LanguageMode0, false);
      Scope scope0 = new Scope(7);
      astRoot0.addChildrenToFront(scope0);
      ErrorCollector errorCollector0 = new ErrorCollector();
      SimpleSourceFile simpleSourceFile0 = new SimpleSourceFile("", true);
      Node node0 = IRFactory.transformTree(astRoot0, simpleSourceFile0, "", config0, errorCollector0);
      assertTrue(node0.isScript());
      assertTrue(node0.isFromExterns());
      assertEquals(0, node0.getLength());
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      TreeSet<String> treeSet0 = new TreeSet<String>();
      WhileLoop whileLoop0 = new WhileLoop();
      astRoot0.addChildrenToFront(whileLoop0);
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5;
      Config config0 = new Config(treeSet0, treeSet0, false, config_LanguageMode0, false);
      ErrorCollector errorCollector0 = new ErrorCollector();
      // Undeclared exception!
      try { 
        IRFactory.transformTree(astRoot0, (StaticSourceFile) null, "$0", config0, errorCollector0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
      }
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      TreeSet<String> treeSet0 = new TreeSet<String>();
      Block block0 = new Block(11);
      astRoot0.addChildrenToFront(block0);
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5_STRICT;
      Config config0 = new Config(treeSet0, treeSet0, false, config_LanguageMode0, false);
      ErrorCollector errorCollector0 = new ErrorCollector();
      Node node0 = IRFactory.transformTree(astRoot0, (StaticSourceFile) null, "language version", config0, errorCollector0);
      assertEquals(1, node0.getChildCount());
      assertTrue(node0.isScript());
      assertEquals(0, node0.getLength());
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      TreeSet<String> treeSet0 = new TreeSet<String>();
      FunctionNode functionNode0 = new FunctionNode();
      astRoot0.addChildrenToFront(functionNode0);
      Name name0 = new Name(5, "Non-JSDoc comment has annotations. Did you mean to start it with '/**'?");
      functionNode0.setFunctionName(name0);
      ErrorCollector errorCollector0 = new ErrorCollector();
      SimpleSourceFile simpleSourceFile0 = new SimpleSourceFile("ciACY*TJEm`!qJc", false);
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT3;
      Config config0 = new Config(treeSet0, treeSet0, false, config_LanguageMode0, false);
      // Undeclared exception!
      try { 
        IRFactory.transformTree(astRoot0, simpleSourceFile0, "Non-JSDoc comment has annotations. Did you mean to start it with '/**'?", config0, errorCollector0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
      }
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      TreeSet<String> treeSet0 = new TreeSet<String>();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5;
      Config config0 = new Config(treeSet0, treeSet0, true, config_LanguageMode0, true);
      Assignment assignment0 = new Assignment();
      astRoot0.addChildrenToFront(assignment0);
      ErrorCollector errorCollector0 = new ErrorCollector();
      // Undeclared exception!
      try { 
        IRFactory.transformTree(astRoot0, (StaticSourceFile) null, (String) null, config0, errorCollector0);
        fail("Expecting exception: UnsupportedOperationException");
      
      } catch(UnsupportedOperationException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.rhino.head.ast.ErrorCollector", e);
      }
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      Locale locale0 = Locale.KOREA;
      Set<String> set0 = locale0.getUnicodeLocaleKeys();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5_STRICT;
      Config config0 = new Config(set0, set0, false, config_LanguageMode0, false);
      ThrowStatement throwStatement0 = new ThrowStatement(astRoot0);
      astRoot0.addChildrenToFront(throwStatement0);
      ErrorCollector errorCollector0 = new ErrorCollector();
      // Undeclared exception!
      try { 
        IRFactory.transformTree(astRoot0, (StaticSourceFile) null, "", config0, errorCollector0);
        fail("Expecting exception: StackOverflowError");
      
      } catch(StackOverflowError e) {
         //
         // no message in exception (getMessage() returned null)
         //
      }
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      TreeSet<String> treeSet0 = new TreeSet<String>();
      RegExpLiteral regExpLiteral0 = new RegExpLiteral(1);
      regExpLiteral0.setValue(" rO4k&b");
      astRoot0.addChildrenToFront(regExpLiteral0);
      SimpleSourceFile simpleSourceFile0 = new SimpleSourceFile(" rO4k&b", false);
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5_STRICT;
      Config config0 = new Config(treeSet0, treeSet0, false, config_LanguageMode0, false);
      ToolErrorReporter toolErrorReporter0 = new ToolErrorReporter(false);
      Node node0 = IRFactory.transformTree(astRoot0, simpleSourceFile0, " rO4k&b", config0, toolErrorReporter0);
      assertEquals(1, node0.getChildCount());
      assertEquals(0, node0.getLength());
      assertTrue(node0.isScript());
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      TreeSet<String> treeSet0 = new TreeSet<String>();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5;
      Config config0 = new Config(treeSet0, treeSet0, true, config_LanguageMode0, true);
      Token.CommentType token_CommentType0 = Token.CommentType.JSDOC;
      Comment comment0 = new Comment(1, 21, token_CommentType0, "wcgke?oW4u_n{");
      astRoot0.addComment(comment0);
      // Undeclared exception!
      try { 
        IRFactory.transformTree(astRoot0, (StaticSourceFile) null, "wcgke?oW4u_n{", config0, (ErrorReporter) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.parsing.JsDocInfoParser$ErrorReporterParser", e);
      }
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      Locale locale0 = Locale.KOREA;
      Set<String> set0 = locale0.getUnicodeLocaleKeys();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5;
      Config config0 = new Config(set0, set0, true, config_LanguageMode0, true);
      Token.CommentType token_CommentType0 = Token.CommentType.JSDOC;
      Comment comment0 = new Comment(1, 2, token_CommentType0, "/Kmafo/");
      astRoot0.setJsDocNode(comment0);
      astRoot0.addComment(comment0);
      ToolErrorReporter toolErrorReporter0 = new ToolErrorReporter(true);
      Node node0 = IRFactory.transformTree(astRoot0, (StaticSourceFile) null, "/Kmafo/", config0, toolErrorReporter0);
      assertEquals(1, node0.getLength());
      assertTrue(node0.isScript());
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      TreeSet<String> treeSet0 = new TreeSet<String>();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5_STRICT;
      Config config0 = new Config(treeSet0, treeSet0, true, config_LanguageMode0, true);
      Token.CommentType token_CommentType0 = Token.CommentType.BLOCK_COMMENT;
      Comment comment0 = new Comment(1, 21, token_CommentType0, "com.google.javascript.jscomp.parsing.IRFactory$TransformDispa@cher");
      astRoot0.addComment(comment0);
      ToolErrorReporter toolErrorReporter0 = new ToolErrorReporter(true);
      Node node0 = IRFactory.transformTree(astRoot0, (StaticSourceFile) null, "com.google.javascript.jscomp.parsing.IRFactory$TransformDispa@cher", config0, toolErrorReporter0);
      assertEquals(1, node0.getLength());
      assertEquals(132, node0.getType());
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      TreeSet<String> treeSet0 = new TreeSet<String>();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5;
      Config config0 = new Config(treeSet0, treeSet0, false, config_LanguageMode0, false);
      Token.CommentType token_CommentType0 = Token.CommentType.BLOCK_COMMENT;
      Comment comment0 = new Comment(1, 0, token_CommentType0, "/* @");
      astRoot0.addComment(comment0);
      ToolErrorReporter toolErrorReporter0 = new ToolErrorReporter(false);
      Node node0 = IRFactory.transformTree(astRoot0, (StaticSourceFile) null, "/* @", config0, toolErrorReporter0);
      assertEquals(0, node0.getLength());
      assertEquals(132, node0.getType());
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      TreeSet<String> treeSet0 = new TreeSet<String>();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5;
      Config config0 = new Config(treeSet0, treeSet0, true, config_LanguageMode0, true);
      ArrayLiteral arrayLiteral0 = new ArrayLiteral(16, 5);
      astRoot0.addChildrenToFront(arrayLiteral0);
      ErrorCollector errorCollector0 = new ErrorCollector();
      Node node0 = IRFactory.transformTree(astRoot0, (StaticSourceFile) null, "\n * @", config0, errorCollector0);
      assertEquals(1, node0.getLength());
      assertEquals(132, node0.getType());
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      TreeSet<String> treeSet0 = new TreeSet<String>();
      ObjectLiteral objectLiteral0 = new ObjectLiteral(22);
      ObjectProperty objectProperty0 = new ObjectProperty(140, (-1565));
      objectLiteral0.addElement(objectProperty0);
      astRoot0.addChildrenToFront(objectLiteral0);
      ErrorCollector errorCollector0 = new ErrorCollector();
      SimpleSourceFile simpleSourceFile0 = new SimpleSourceFile("$0", true);
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5_STRICT;
      Config config0 = new Config(treeSet0, treeSet0, true, config_LanguageMode0, true);
      // Undeclared exception!
      try { 
        IRFactory.transformTree(astRoot0, simpleSourceFile0, "$0", config0, errorCollector0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
      }
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      TreeSet<String> treeSet0 = new TreeSet<String>();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5;
      Config config0 = new Config(treeSet0, treeSet0, false, config_LanguageMode0, false);
      ArrayLiteral arrayLiteral0 = new ArrayLiteral();
      arrayLiteral0.addElement(astRoot0);
      astRoot0.addChildrenToFront(arrayLiteral0);
      ErrorCollector errorCollector0 = new ErrorCollector();
      SimpleSourceFile simpleSourceFile0 = new SimpleSourceFile("\"p<:Ze!", false);
      // Undeclared exception!
      try { 
        IRFactory.transformTree(astRoot0, simpleSourceFile0, "\"p<:Ze!", config0, errorCollector0);
        fail("Expecting exception: StackOverflowError");
      
      } catch(StackOverflowError e) {
         //
         // no message in exception (getMessage() returned null)
         //
      }
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      TreeSet<String> treeSet0 = new TreeSet<String>();
      FunctionNode functionNode0 = new FunctionNode(0);
      astRoot0.addChildrenToFront(functionNode0);
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5_STRICT;
      Config config0 = new Config(treeSet0, treeSet0, true, config_LanguageMode0, true);
      ToolErrorReporter toolErrorReporter0 = new ToolErrorReporter(true);
      Node node0 = IRFactory.transformTree(astRoot0, (StaticSourceFile) null, "freeze", config0, toolErrorReporter0);
      assertEquals(1, node0.getLength());
      assertTrue(node0.isScript());
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      TreeSet<String> treeSet0 = new TreeSet<String>();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT3;
      Config config0 = new Config(treeSet0, treeSet0, true, config_LanguageMode0, true);
      BreakStatement breakStatement0 = new BreakStatement();
      astRoot0.addChildrenToFront(breakStatement0);
      ErrorCollector errorCollector0 = new ErrorCollector();
      Node node0 = IRFactory.transformTree(astRoot0, (StaticSourceFile) null, "+:-HWFpG4(md", config0, errorCollector0);
      assertTrue(node0.isScript());
      assertEquals(1, node0.getLength());
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      TreeSet<String> treeSet0 = new TreeSet<String>();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT3;
      Config config0 = new Config(treeSet0, treeSet0, true, config_LanguageMode0, true);
      ContinueStatement continueStatement0 = new ContinueStatement(2, 23);
      astRoot0.addChildrenToFront(continueStatement0);
      ErrorCollector errorCollector0 = new ErrorCollector();
      Node node0 = IRFactory.transformTree(astRoot0, (StaticSourceFile) null, "f:fVd", config0, errorCollector0);
      assertEquals(1, node0.getLength());
      assertEquals(132, node0.getType());
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      Name name0 = new Name(6, 25, "unnamed function statement");
      ContinueStatement continueStatement0 = new ContinueStatement(name0);
      astRoot0.addChild(continueStatement0);
      TreeSet<String> treeSet0 = new TreeSet<String>();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5_STRICT;
      Config config0 = new Config(treeSet0, treeSet0, true, config_LanguageMode0, true);
      ToolErrorReporter toolErrorReporter0 = new ToolErrorReporter(true);
      Node node0 = IRFactory.transformTree(astRoot0, (StaticSourceFile) null, "interface", config0, toolErrorReporter0);
      assertTrue(node0.isScript());
      assertEquals(1, node0.getLength());
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      TreeSet<String> treeSet0 = new TreeSet<String>();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5_STRICT;
      Config config0 = new Config(treeSet0, treeSet0, false, config_LanguageMode0, false);
      FunctionNode functionNode0 = new FunctionNode(1);
      astRoot0.addChildrenToFront(functionNode0);
      functionNode0.addParam(astRoot0);
      Name name0 = new Name(15, "Non-JSDoc comment has annotations. Did you mean to start it with '/**'?");
      functionNode0.setFunctionName(name0);
      ErrorCollector errorCollector0 = new ErrorCollector();
      SimpleSourceFile simpleSourceFile0 = new SimpleSourceFile("unnamed function statement", false);
      // Undeclared exception!
      IRFactory.transformTree(astRoot0, simpleSourceFile0, "unnamed function statement", config0, errorCollector0);
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      LinkedHashSet<String> linkedHashSet0 = new LinkedHashSet<String>();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5_STRICT;
      Config config0 = new Config(linkedHashSet0, linkedHashSet0, false, config_LanguageMode0, false);
      LabeledStatement labeledStatement0 = new LabeledStatement();
      ReturnStatement returnStatement0 = new ReturnStatement(0, 1, labeledStatement0);
      astRoot0.addChildrenToFront(returnStatement0);
      ErrorCollector errorCollector0 = new ErrorCollector();
      // Undeclared exception!
      try { 
        IRFactory.transformTree(astRoot0, (StaticSourceFile) null, (String) null, config0, errorCollector0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
      }
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      TreeSet<String> treeSet0 = new TreeSet<String>();
      ObjectLiteral objectLiteral0 = new ObjectLiteral();
      astRoot0.addChildrenToFront(objectLiteral0);
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5;
      Config config0 = new Config(treeSet0, treeSet0, true, config_LanguageMode0, true);
      ErrorCollector errorCollector0 = new ErrorCollector();
      SimpleSourceFile simpleSourceFile0 = new SimpleSourceFile("S", true);
      Node node0 = IRFactory.transformTree(astRoot0, simpleSourceFile0, "S", config0, errorCollector0);
      assertTrue(node0.isFromExterns());
      assertEquals(1, node0.getLength());
      assertTrue(node0.isScript());
  }

  @Test(timeout = 4000)
  public void test35()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      TreeSet<String> treeSet0 = new TreeSet<String>();
      ObjectLiteral objectLiteral0 = new ObjectLiteral(22);
      ObjectProperty objectProperty0 = new ObjectProperty(140, (-1576));
      objectLiteral0.addElement(objectProperty0);
      astRoot0.addChildrenToFront(objectLiteral0);
      ErrorCollector errorCollector0 = new ErrorCollector();
      SimpleSourceFile simpleSourceFile0 = new SimpleSourceFile("$0", true);
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT3;
      Config config0 = new Config(treeSet0, treeSet0, true, config_LanguageMode0, true);
      // Undeclared exception!
      try { 
        IRFactory.transformTree(astRoot0, simpleSourceFile0, "$0", config0, errorCollector0);
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
      TreeSet<String> treeSet0 = new TreeSet<String>();
      ReturnStatement returnStatement0 = new ReturnStatement();
      astRoot0.addChildrenToFront(returnStatement0);
      ErrorCollector errorCollector0 = new ErrorCollector();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT3;
      Config config0 = new Config(treeSet0, treeSet0, false, config_LanguageMode0, false);
      Node node0 = IRFactory.transformTree(astRoot0, (StaticSourceFile) null, "getters may not have parameters", config0, errorCollector0);
      assertEquals(0, node0.getLength());
      assertTrue(node0.isScript());
  }

  @Test(timeout = 4000)
  public void test37()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      Locale locale0 = Locale.KOREA;
      Set<String> set0 = locale0.getUnicodeLocaleKeys();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT3;
      SwitchCase switchCase0 = new SwitchCase();
      switchCase0.setExpression(astRoot0);
      astRoot0.addChildrenToFront(switchCase0);
      ErrorCollector errorCollector0 = new ErrorCollector();
      Config config0 = new Config(set0, set0, true, config_LanguageMode0, true);
      // Undeclared exception!
      try { 
        IRFactory.transformTree(astRoot0, (StaticSourceFile) null, "private", config0, errorCollector0);
        fail("Expecting exception: StackOverflowError");
      
      } catch(StackOverflowError e) {
      }
  }

  @Test(timeout = 4000)
  public void test38()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      TreeSet<String> treeSet0 = new TreeSet<String>();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5;
      Config config0 = new Config(treeSet0, treeSet0, false, config_LanguageMode0, false);
      SwitchCase switchCase0 = new SwitchCase(23, 24);
      astRoot0.addChildrenToFront(switchCase0);
      ErrorCollector errorCollector0 = new ErrorCollector();
      Node node0 = IRFactory.transformTree(astRoot0, (StaticSourceFile) null, "", config0, errorCollector0);
      assertEquals(0, node0.getLength());
  }

  @Test(timeout = 4000)
  public void test39()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      TreeSet<String> treeSet0 = new TreeSet<String>();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5;
      Config config0 = new Config(treeSet0, treeSet0, true, config_LanguageMode0, true);
      VariableDeclaration variableDeclaration0 = new VariableDeclaration(0, 2);
      astRoot0.addChildrenToFront(variableDeclaration0);
      ErrorCollector errorCollector0 = new ErrorCollector();
      Node node0 = IRFactory.transformTree(astRoot0, (StaticSourceFile) null, "unnamed function statement", config0, errorCollector0);
      assertEquals(1, node0.getLength());
  }

  @Test(timeout = 4000)
  public void test40()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      TreeSet<String> treeSet0 = new TreeSet<String>();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT3;
      Config config0 = new Config(treeSet0, treeSet0, false, config_LanguageMode0, false);
      VariableDeclaration variableDeclaration0 = new VariableDeclaration(21, 4);
      astRoot0.addChildrenToFront(variableDeclaration0);
      ErrorCollector errorCollector0 = new ErrorCollector();
      SimpleSourceFile simpleSourceFile0 = new SimpleSourceFile("JSDOC", false);
      Node node0 = IRFactory.transformTree(astRoot0, simpleSourceFile0, "JSDOC", config0, errorCollector0);
      assertEquals(0, node0.getLength());
  }

  @Test(timeout = 4000)
  public void test41()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      TreeSet<String> treeSet0 = new TreeSet<String>();
      FunctionCall functionCall0 = new FunctionCall(140, 2);
      astRoot0.addChildrenToFront(functionCall0);
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5_STRICT;
      Config config0 = new Config(treeSet0, treeSet0, false, config_LanguageMode0, false);
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
}
