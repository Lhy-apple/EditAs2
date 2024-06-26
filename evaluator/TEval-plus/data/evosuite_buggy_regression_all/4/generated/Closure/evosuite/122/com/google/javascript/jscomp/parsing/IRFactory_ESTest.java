/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 21:22:05 GMT 2023
 */

package com.google.javascript.jscomp.parsing;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.google.javascript.jscomp.parsing.Config;
import com.google.javascript.jscomp.parsing.IRFactory;
import com.google.javascript.rhino.JSDocInfo;
import com.google.javascript.rhino.Node;
import com.google.javascript.rhino.head.Context;
import com.google.javascript.rhino.head.ContextFactory;
import com.google.javascript.rhino.head.ErrorReporter;
import com.google.javascript.rhino.head.Token;
import com.google.javascript.rhino.head.ast.ArrayLiteral;
import com.google.javascript.rhino.head.ast.AstRoot;
import com.google.javascript.rhino.head.ast.Block;
import com.google.javascript.rhino.head.ast.Comment;
import com.google.javascript.rhino.head.ast.ConditionalExpression;
import com.google.javascript.rhino.head.ast.ContinueStatement;
import com.google.javascript.rhino.head.ast.DoLoop;
import com.google.javascript.rhino.head.ast.ElementGet;
import com.google.javascript.rhino.head.ast.EmptyExpression;
import com.google.javascript.rhino.head.ast.EmptyStatement;
import com.google.javascript.rhino.head.ast.ErrorCollector;
import com.google.javascript.rhino.head.ast.ExpressionStatement;
import com.google.javascript.rhino.head.ast.ForLoop;
import com.google.javascript.rhino.head.ast.FunctionCall;
import com.google.javascript.rhino.head.ast.FunctionNode;
import com.google.javascript.rhino.head.ast.GeneratorExpressionLoop;
import com.google.javascript.rhino.head.ast.Label;
import com.google.javascript.rhino.head.ast.LabeledStatement;
import com.google.javascript.rhino.head.ast.Name;
import com.google.javascript.rhino.head.ast.NewExpression;
import com.google.javascript.rhino.head.ast.NumberLiteral;
import com.google.javascript.rhino.head.ast.ObjectLiteral;
import com.google.javascript.rhino.head.ast.ObjectProperty;
import com.google.javascript.rhino.head.ast.ParenthesizedExpression;
import com.google.javascript.rhino.head.ast.ReturnStatement;
import com.google.javascript.rhino.head.ast.StringLiteral;
import com.google.javascript.rhino.head.ast.SwitchCase;
import com.google.javascript.rhino.head.ast.ThrowStatement;
import com.google.javascript.rhino.head.ast.VariableDeclaration;
import com.google.javascript.rhino.head.ast.VariableInitializer;
import com.google.javascript.rhino.head.ast.WhileLoop;
import com.google.javascript.rhino.head.ast.WithStatement;
import com.google.javascript.rhino.head.ast.XmlExpression;
import com.google.javascript.rhino.head.tools.ToolErrorReporter;
import com.google.javascript.rhino.jstype.SimpleSourceFile;
import com.google.javascript.rhino.jstype.StaticSourceFile;
import java.nio.charset.Charset;
import java.util.HashSet;
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
      ConditionalExpression conditionalExpression0 = new ConditionalExpression(11);
      astRoot0.addChildrenToBack(conditionalExpression0);
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT3;
      Config config0 = new Config(treeSet0, treeSet0, true, config_LanguageMode0, true);
      ToolErrorReporter toolErrorReporter0 = new ToolErrorReporter(true);
      // Undeclared exception!
      try { 
        IRFactory.transformTree(astRoot0, (StaticSourceFile) null, "9f{", config0, toolErrorReporter0);
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
      SimpleSourceFile simpleSourceFile0 = new SimpleSourceFile("", false);
      TreeSet<String> treeSet0 = new TreeSet<String>();
      ParenthesizedExpression parenthesizedExpression0 = new ParenthesizedExpression((-534), 200000);
      astRoot0.addChildrenToBack(parenthesizedExpression0);
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT3;
      Config config0 = new Config(treeSet0, treeSet0, false, config_LanguageMode0, false);
      ToolErrorReporter toolErrorReporter0 = new ToolErrorReporter(false);
      // Undeclared exception!
      try { 
        IRFactory.transformTree(astRoot0, simpleSourceFile0, "K7`6/l`Ew", config0, toolErrorReporter0);
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
      SimpleSourceFile simpleSourceFile0 = new SimpleSourceFile("K` /l`w", true);
      TreeSet<String> treeSet0 = new TreeSet<String>();
      DoLoop doLoop0 = new DoLoop(37);
      astRoot0.addChildrenToBack(doLoop0);
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT3;
      Config config0 = new Config(treeSet0, treeSet0, true, config_LanguageMode0, false);
      ToolErrorReporter toolErrorReporter0 = new ToolErrorReporter(true);
      // Undeclared exception!
      try { 
        IRFactory.transformTree(astRoot0, simpleSourceFile0, "pL;V`4^WQ !Mr^+H: H", config0, toolErrorReporter0);
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
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5;
      Config config0 = new Config(treeSet0, treeSet0, false, config_LanguageMode0, false);
      ToolErrorReporter toolErrorReporter0 = new ToolErrorReporter(false);
      EmptyExpression emptyExpression0 = new EmptyExpression(19);
      astRoot0.addChildrenToBack(emptyExpression0);
      SimpleSourceFile simpleSourceFile0 = new SimpleSourceFile("^3cFwOwk4JaCX", false);
      Node node0 = IRFactory.transformTree(astRoot0, simpleSourceFile0, "^3cFwOwk4JaCX", config0, toolErrorReporter0);
      assertTrue(node0.isScript());
      assertEquals(0, node0.getLength());
      assertTrue(node0.hasOneChild());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      TreeSet<String> treeSet0 = new TreeSet<String>();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5;
      Config config0 = new Config(treeSet0, treeSet0, false, config_LanguageMode0, false);
      ToolErrorReporter toolErrorReporter0 = new ToolErrorReporter(false);
      ElementGet elementGet0 = new ElementGet();
      astRoot0.addChildrenToBack(elementGet0);
      // Undeclared exception!
      try { 
        IRFactory.transformTree(astRoot0, (StaticSourceFile) null, "language version", config0, toolErrorReporter0);
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
      TreeSet<String> treeSet0 = new TreeSet<String>();
      ObjectProperty objectProperty0 = new ObjectProperty(20);
      astRoot0.addChildrenToBack(objectProperty0);
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5_STRICT;
      Config config0 = new Config(treeSet0, treeSet0, true, config_LanguageMode0, true);
      ToolErrorReporter toolErrorReporter0 = new ToolErrorReporter(true);
      // Undeclared exception!
      try { 
        IRFactory.transformTree(astRoot0, (StaticSourceFile) null, " p_=ouwQW+cy", config0, toolErrorReporter0);
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // 103
         //
         verifyException("com.google.javascript.jscomp.parsing.IRFactory", e);
      }
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      TreeSet<String> treeSet0 = new TreeSet<String>();
      WithStatement withStatement0 = new WithStatement(26);
      astRoot0.addChildrenToBack(withStatement0);
      SimpleSourceFile simpleSourceFile0 = new SimpleSourceFile("", true);
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5_STRICT;
      Config config0 = new Config(treeSet0, treeSet0, true, config_LanguageMode0, true);
      ToolErrorReporter toolErrorReporter0 = new ToolErrorReporter(false);
      // Undeclared exception!
      try { 
        IRFactory.transformTree(astRoot0, simpleSourceFile0, "l*-dOG~Z;@", config0, toolErrorReporter0);
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
      ForLoop forLoop0 = new ForLoop(4, 3222);
      astRoot0.addChildrenToBack(forLoop0);
      Locale locale0 = Locale.ITALIAN;
      Set<String> set0 = locale0.getUnicodeLocaleAttributes();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5_STRICT;
      Config config0 = new Config(set0, set0, false, config_LanguageMode0, false);
      ToolErrorReporter toolErrorReporter0 = new ToolErrorReporter(false);
      // Undeclared exception!
      try { 
        IRFactory.transformTree(astRoot0, (StaticSourceFile) null, "", config0, toolErrorReporter0);
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
      SimpleSourceFile simpleSourceFile0 = new SimpleSourceFile("K` /l`w", true);
      Charset charset0 = Charset.defaultCharset();
      Set<String> set0 = charset0.aliases();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT3;
      Config config0 = new Config(set0, set0, false, config_LanguageMode0, true);
      ErrorCollector errorCollector0 = new ErrorCollector();
      Label label0 = new Label(8, (-754));
      astRoot0.addChildToFront(label0);
      // Undeclared exception!
      try { 
        IRFactory.transformTree(astRoot0, simpleSourceFile0, "invalid decrement target", config0, errorCollector0);
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
      NumberLiteral numberLiteral0 = new NumberLiteral(1.0);
      astRoot0.addChildrenToBack(numberLiteral0);
      HashSet<String> hashSet0 = new HashSet<String>();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5_STRICT;
      Config config0 = new Config(hashSet0, hashSet0, true, config_LanguageMode0, false);
      ErrorCollector errorCollector0 = new ErrorCollector();
      Node node0 = IRFactory.transformTree(astRoot0, (StaticSourceFile) null, "", config0, errorCollector0);
      assertEquals(1, node0.getLength());
      assertEquals(132, node0.getType());
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      TreeSet<String> treeSet0 = new TreeSet<String>();
      ExpressionStatement expressionStatement0 = new ExpressionStatement();
      astRoot0.addChildrenToBack(expressionStatement0);
      ToolErrorReporter toolErrorReporter0 = new ToolErrorReporter(false);
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5_STRICT;
      Config config0 = new Config(treeSet0, treeSet0, false, config_LanguageMode0, false);
      // Undeclared exception!
      try { 
        IRFactory.transformTree(astRoot0, (StaticSourceFile) null, "", config0, toolErrorReporter0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
      }
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      TreeSet<String> treeSet0 = new TreeSet<String>();
      EmptyStatement emptyStatement0 = new EmptyStatement(3260);
      astRoot0.addChildrenToBack(emptyStatement0);
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5_STRICT;
      Config config0 = new Config(treeSet0, treeSet0, false, config_LanguageMode0, true);
      ToolErrorReporter toolErrorReporter0 = new ToolErrorReporter(false);
      Node node0 = IRFactory.transformTree(astRoot0, (StaticSourceFile) null, "implements", config0, toolErrorReporter0);
      assertEquals(0, node0.getLength());
      assertEquals(132, node0.getType());
      assertEquals(1, node0.getChildCount());
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      SimpleSourceFile simpleSourceFile0 = new SimpleSourceFile("[B`6/`N", false);
      TreeSet<String> treeSet0 = new TreeSet<String>();
      WhileLoop whileLoop0 = new WhileLoop((-196), 3);
      astRoot0.addChildrenToBack(whileLoop0);
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5;
      Config config0 = new Config(treeSet0, treeSet0, false, config_LanguageMode0, true);
      ToolErrorReporter toolErrorReporter0 = new ToolErrorReporter(true);
      // Undeclared exception!
      try { 
        IRFactory.transformTree(astRoot0, simpleSourceFile0, "[B`6/`N", config0, toolErrorReporter0);
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
      SimpleSourceFile simpleSourceFile0 = new SimpleSourceFile("K` /l`w", true);
      Block block0 = new Block(150, 23);
      astRoot0.addChildrenToBack(block0);
      LinkedHashSet<String> linkedHashSet0 = new LinkedHashSet<String>();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5;
      Config config0 = new Config(linkedHashSet0, linkedHashSet0, true, config_LanguageMode0, true);
      ToolErrorReporter toolErrorReporter0 = new ToolErrorReporter(true);
      Node node0 = IRFactory.transformTree(astRoot0, simpleSourceFile0, "'4+xxVSMB+o&ckz", config0, toolErrorReporter0);
      assertTrue(node0.isFromExterns());
      assertEquals(1, node0.getLength());
      assertTrue(node0.isScript());
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      SimpleSourceFile simpleSourceFile0 = new SimpleSourceFile("implements", true);
      TreeSet<String> treeSet0 = new TreeSet<String>();
      Name name0 = new Name(88, 21, "implements");
      ContinueStatement continueStatement0 = new ContinueStatement(3894, 8, name0);
      astRoot0.addChildrenToBack(continueStatement0);
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5_STRICT;
      Config config0 = new Config(treeSet0, treeSet0, true, config_LanguageMode0, true);
      ContextFactory contextFactory0 = new ContextFactory();
      Context context0 = contextFactory0.enterContext();
      ErrorReporter errorReporter0 = context0.getErrorReporter();
      // Undeclared exception!
      try { 
        IRFactory.transformTree(astRoot0, simpleSourceFile0, "", config0, errorReporter0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // identifier is a reserved word
         //
         verifyException("com.google.javascript.rhino.head.DefaultErrorReporter", e);
      }
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      TreeSet<String> treeSet0 = new TreeSet<String>();
      XmlExpression xmlExpression0 = new XmlExpression(140, 10);
      astRoot0.addChildrenToBack(xmlExpression0);
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5_STRICT;
      Config config0 = new Config(treeSet0, treeSet0, false, config_LanguageMode0, false);
      ToolErrorReporter toolErrorReporter0 = new ToolErrorReporter(false);
      Node node0 = IRFactory.transformTree(astRoot0, (StaticSourceFile) null, "]\n", config0, toolErrorReporter0);
      assertEquals(0, node0.getLength());
      assertTrue(node0.isScript());
      assertTrue(node0.hasOneChild());
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      SimpleSourceFile simpleSourceFile0 = new SimpleSourceFile("K` /l`w", true);
      TreeSet<String> treeSet0 = new TreeSet<String>();
      ThrowStatement throwStatement0 = new ThrowStatement(astRoot0);
      astRoot0.addChildrenToBack(throwStatement0);
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5;
      Config config0 = new Config(treeSet0, treeSet0, false, config_LanguageMode0, false);
      ErrorCollector errorCollector0 = new ErrorCollector();
      // Undeclared exception!
      try { 
        IRFactory.transformTree(astRoot0, simpleSourceFile0, "K` /l`w", config0, errorCollector0);
        fail("Expecting exception: StackOverflowError");
      
      } catch(StackOverflowError e) {
         //
         // no message in exception (getMessage() returned null)
         //
      }
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      TreeSet<String> treeSet0 = new TreeSet<String>();
      FunctionNode functionNode0 = new FunctionNode(9);
      astRoot0.addChildrenToBack(functionNode0);
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5;
      Config config0 = new Config(treeSet0, treeSet0, false, config_LanguageMode0, false);
      ToolErrorReporter toolErrorReporter0 = new ToolErrorReporter(false);
      Node node0 = IRFactory.transformTree(astRoot0, (StaticSourceFile) null, "", config0, toolErrorReporter0);
      assertEquals(0, node0.getLength());
      assertTrue(node0.isScript());
      assertEquals(1, node0.getChildCount());
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      Token.CommentType token_CommentType0 = Token.CommentType.JSDOC;
      Comment comment0 = new Comment(15, 1, token_CommentType0, "");
      astRoot0.addComment(comment0);
      ToolErrorReporter toolErrorReporter0 = new ToolErrorReporter(true);
      SimpleSourceFile simpleSourceFile0 = new SimpleSourceFile("", true);
      Charset charset0 = Charset.defaultCharset();
      Set<String> set0 = charset0.aliases();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5_STRICT;
      Config config0 = new Config(set0, set0, true, config_LanguageMode0, true);
      // Undeclared exception!
      try { 
        IRFactory.transformTree(astRoot0, simpleSourceFile0, "", config0, toolErrorReporter0);
        fail("Expecting exception: StringIndexOutOfBoundsException");
      
      } catch(StringIndexOutOfBoundsException e) {
      }
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      TreeSet<String> treeSet0 = new TreeSet<String>();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5_STRICT;
      Config config0 = new Config(treeSet0, treeSet0, false, config_LanguageMode0, false);
      ToolErrorReporter toolErrorReporter0 = new ToolErrorReporter(false);
      StringLiteral stringLiteral0 = new StringLiteral(1);
      astRoot0.addChildrenToBack(stringLiteral0);
      // Undeclared exception!
      try { 
        IRFactory.transformTree(astRoot0, (StaticSourceFile) null, "<;3Q<8#zQ&#-O", config0, toolErrorReporter0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // StringNode: str is null
         //
         verifyException("com.google.javascript.rhino.Node$StringNode", e);
      }
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      Token.CommentType token_CommentType0 = Token.CommentType.BLOCK_COMMENT;
      Comment comment0 = new Comment(15, 1, token_CommentType0, "");
      astRoot0.addComment(comment0);
      ToolErrorReporter toolErrorReporter0 = new ToolErrorReporter(true);
      SimpleSourceFile simpleSourceFile0 = new SimpleSourceFile("", true);
      Charset charset0 = Charset.defaultCharset();
      Set<String> set0 = charset0.aliases();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5_STRICT;
      Config config0 = new Config(set0, set0, true, config_LanguageMode0, true);
      Node node0 = IRFactory.transformTree(astRoot0, simpleSourceFile0, "", config0, toolErrorReporter0);
      assertEquals(1, node0.getLength());
      assertEquals(132, node0.getType());
      assertTrue(node0.isFromExterns());
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      Token.CommentType token_CommentType0 = Token.CommentType.LINE;
      Comment comment0 = new Comment(15, 1, token_CommentType0, "");
      astRoot0.addComment(comment0);
      ToolErrorReporter toolErrorReporter0 = new ToolErrorReporter(true);
      SimpleSourceFile simpleSourceFile0 = new SimpleSourceFile("", true);
      Charset charset0 = Charset.defaultCharset();
      Set<String> set0 = charset0.aliases();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5_STRICT;
      Config config0 = new Config(set0, set0, true, config_LanguageMode0, true);
      Node node0 = IRFactory.transformTree(astRoot0, simpleSourceFile0, "", config0, toolErrorReporter0);
      assertTrue(node0.isFromExterns());
      assertEquals(132, node0.getType());
      assertEquals(1, node0.getLength());
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      SimpleSourceFile simpleSourceFile0 = new SimpleSourceFile("K` /l`w", false);
      TreeSet<String> treeSet0 = new TreeSet<String>();
      Token.CommentType token_CommentType0 = Token.CommentType.LINE;
      Comment comment0 = new Comment(23, 4, token_CommentType0, "K` /l`w");
      astRoot0.setJsDocNode(comment0);
      ToolErrorReporter toolErrorReporter0 = new ToolErrorReporter(false);
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5_STRICT;
      Config config0 = new Config(treeSet0, treeSet0, false, config_LanguageMode0, false);
      Node node0 = IRFactory.transformTree(astRoot0, simpleSourceFile0, "K` /l`w", config0, toolErrorReporter0);
      assertTrue(node0.isScript());
      assertEquals(0, node0.getLength());
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      JSDocInfo jSDocInfo0 = new JSDocInfo();
      Set<String> set0 = jSDocInfo0.getSuppressions();
      ArrayLiteral arrayLiteral0 = new ArrayLiteral(13);
      astRoot0.addChildrenToBack(arrayLiteral0);
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT3;
      Config config0 = new Config(set0, set0, false, config_LanguageMode0, false);
      ToolErrorReporter toolErrorReporter0 = new ToolErrorReporter(false);
      Node node0 = IRFactory.transformTree(astRoot0, (StaticSourceFile) null, "let", config0, toolErrorReporter0);
      assertEquals(0, node0.getLength());
      assertEquals(132, node0.getType());
      assertTrue(node0.hasChildren());
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      TreeSet<String> treeSet0 = new TreeSet<String>();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5;
      Config config0 = new Config(treeSet0, treeSet0, false, config_LanguageMode0, false);
      ToolErrorReporter toolErrorReporter0 = new ToolErrorReporter(false);
      ContinueStatement continueStatement0 = new ContinueStatement();
      astRoot0.addChildrenToBack(continueStatement0);
      Node node0 = IRFactory.transformTree(astRoot0, (StaticSourceFile) null, "let", config0, toolErrorReporter0);
      assertEquals(0, node0.getLength());
      assertEquals(132, node0.getType());
      assertTrue(node0.hasOneChild());
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      SimpleSourceFile simpleSourceFile0 = new SimpleSourceFile("K` /l`w", true);
      TreeSet<String> treeSet0 = new TreeSet<String>();
      GeneratorExpressionLoop generatorExpressionLoop0 = new GeneratorExpressionLoop(6);
      astRoot0.addChildrenToBack(generatorExpressionLoop0);
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT3;
      Config config0 = new Config(treeSet0, treeSet0, true, config_LanguageMode0, true);
      ToolErrorReporter toolErrorReporter0 = new ToolErrorReporter(true);
      // Undeclared exception!
      try { 
        IRFactory.transformTree(astRoot0, simpleSourceFile0, "K` /l`w", config0, toolErrorReporter0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
      }
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      SimpleSourceFile simpleSourceFile0 = new SimpleSourceFile("?.Oo", true);
      TreeSet<String> treeSet0 = new TreeSet<String>();
      LabeledStatement labeledStatement0 = new LabeledStatement();
      astRoot0.addChildrenToBack(labeledStatement0);
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5_STRICT;
      Config config0 = new Config(treeSet0, treeSet0, true, config_LanguageMode0, true);
      ToolErrorReporter toolErrorReporter0 = new ToolErrorReporter(true);
      // Undeclared exception!
      try { 
        IRFactory.transformTree(astRoot0, simpleSourceFile0, "?.Oo", config0, toolErrorReporter0);
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
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5_STRICT;
      Config config0 = new Config(treeSet0, treeSet0, false, config_LanguageMode0, false);
      ToolErrorReporter toolErrorReporter0 = new ToolErrorReporter(false);
      Name name0 = new Name(23, 2);
      ContinueStatement continueStatement0 = new ContinueStatement(name0);
      astRoot0.addChildrenToBack(continueStatement0);
      // Undeclared exception!
      try { 
        IRFactory.transformTree(astRoot0, (StaticSourceFile) null, " p_=ouwQW+cy", config0, toolErrorReporter0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // StringNode: str is null
         //
         verifyException("com.google.javascript.rhino.Node$StringNode", e);
      }
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      TreeSet<String> treeSet0 = new TreeSet<String>();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT3;
      Config config0 = new Config(treeSet0, treeSet0, false, config_LanguageMode0, false);
      ToolErrorReporter toolErrorReporter0 = new ToolErrorReporter(false);
      Name name0 = new Name(23, 2);
      astRoot0.addChildrenToBack(name0);
      // Undeclared exception!
      try { 
        IRFactory.transformTree(astRoot0, (StaticSourceFile) null, "?pu=ouwQW+cy", config0, toolErrorReporter0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.rhino.TokenStream", e);
      }
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      TreeSet<String> treeSet0 = new TreeSet<String>();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5_STRICT;
      Config config0 = new Config(treeSet0, treeSet0, true, config_LanguageMode0, true);
      ToolErrorReporter toolErrorReporter0 = new ToolErrorReporter(true);
      ObjectLiteral objectLiteral0 = new ObjectLiteral(1);
      astRoot0.addChildrenToBack(objectLiteral0);
      Node node0 = IRFactory.transformTree(astRoot0, (StaticSourceFile) null, "unsupported language extension: for each", config0, toolErrorReporter0);
      assertEquals(1, node0.getLength());
      assertTrue(node0.isScript());
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      SimpleSourceFile simpleSourceFile0 = new SimpleSourceFile("K7`6/l`w", true);
      TreeSet<String> treeSet0 = new TreeSet<String>();
      ReturnStatement returnStatement0 = new ReturnStatement();
      astRoot0.addChildrenToBack(returnStatement0);
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5;
      Config config0 = new Config(treeSet0, treeSet0, true, config_LanguageMode0, true);
      ErrorCollector errorCollector0 = new ErrorCollector();
      Node node0 = IRFactory.transformTree(astRoot0, simpleSourceFile0, "K7`6/l`w", config0, errorCollector0);
      assertEquals(1, node0.getLength());
      assertEquals(132, node0.getType());
      assertTrue(node0.isFromExterns());
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      SimpleSourceFile simpleSourceFile0 = new SimpleSourceFile("", true);
      TreeSet<String> treeSet0 = new TreeSet<String>();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5;
      Config config0 = new Config(treeSet0, treeSet0, true, config_LanguageMode0, true);
      ReturnStatement returnStatement0 = new ReturnStatement(74, 3, astRoot0);
      astRoot0.addChildToFront(returnStatement0);
      ContextFactory contextFactory0 = ContextFactory.getGlobal();
      Context context0 = contextFactory0.enterContext();
      ErrorReporter errorReporter0 = context0.getErrorReporter();
      // Undeclared exception!
      try { 
        IRFactory.transformTree(astRoot0, simpleSourceFile0, "", config0, errorReporter0);
        fail("Expecting exception: StackOverflowError");
      
      } catch(StackOverflowError e) {
         //
         // no message in exception (getMessage() returned null)
         //
      }
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      SimpleSourceFile simpleSourceFile0 = new SimpleSourceFile("K7`6/l`w", false);
      TreeSet<String> treeSet0 = new TreeSet<String>();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5;
      Config config0 = new Config(treeSet0, treeSet0, false, config_LanguageMode0, false);
      ToolErrorReporter toolErrorReporter0 = new ToolErrorReporter(false);
      SwitchCase switchCase0 = new SwitchCase(4);
      astRoot0.addChildrenToBack(switchCase0);
      Node node0 = IRFactory.transformTree(astRoot0, simpleSourceFile0, "K7`6/l`w", config0, toolErrorReporter0);
      assertEquals(0, node0.getLength());
      assertTrue(node0.hasOneChild());
      assertTrue(node0.isScript());
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      SimpleSourceFile simpleSourceFile0 = new SimpleSourceFile("K` /l`w", true);
      TreeSet<String> treeSet0 = new TreeSet<String>();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT3;
      Config config0 = new Config(treeSet0, treeSet0, false, config_LanguageMode0, false);
      ToolErrorReporter toolErrorReporter0 = new ToolErrorReporter(true);
      SwitchCase switchCase0 = new SwitchCase(2095);
      switchCase0.addStatement(astRoot0);
      astRoot0.addChildrenToBack(switchCase0);
      // Undeclared exception!
      try { 
        IRFactory.transformTree(astRoot0, simpleSourceFile0, "MR,q6U_", config0, toolErrorReporter0);
        fail("Expecting exception: StackOverflowError");
      
      } catch(StackOverflowError e) {
         //
         // no message in exception (getMessage() returned null)
         //
      }
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      TreeSet<String> treeSet0 = new TreeSet<String>();
      VariableDeclaration variableDeclaration0 = new VariableDeclaration(21, 2);
      astRoot0.addChildrenToBack(variableDeclaration0);
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT3;
      Config config0 = new Config(treeSet0, treeSet0, true, config_LanguageMode0, true);
      Context context0 = new Context();
      ErrorReporter errorReporter0 = context0.getErrorReporter();
      Node node0 = IRFactory.transformTree(astRoot0, (StaticSourceFile) null, "error reporter", config0, errorReporter0);
      assertEquals(1, node0.getLength());
      assertEquals(132, node0.getType());
  }

  @Test(timeout = 4000)
  public void test35()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      TreeSet<String> treeSet0 = new TreeSet<String>();
      VariableDeclaration variableDeclaration0 = new VariableDeclaration(21, 2);
      astRoot0.addChildrenToBack(variableDeclaration0);
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT3;
      Config config0 = new Config(treeSet0, treeSet0, false, config_LanguageMode0, false);
      Context context0 = new Context();
      ErrorReporter errorReporter0 = context0.getErrorReporter();
      Node node0 = IRFactory.transformTree(astRoot0, (StaticSourceFile) null, "error reporter", config0, errorReporter0);
      assertTrue(node0.hasChildren());
      assertTrue(node0.isScript());
      assertEquals(0, node0.getLength());
  }

  @Test(timeout = 4000)
  public void test36()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      TreeSet<String> treeSet0 = new TreeSet<String>();
      VariableDeclaration variableDeclaration0 = new VariableDeclaration(6);
      VariableInitializer variableInitializer0 = new VariableInitializer();
      variableDeclaration0.addVariable(variableInitializer0);
      astRoot0.addChildrenToBack(variableDeclaration0);
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT3;
      Config config0 = new Config(treeSet0, treeSet0, false, config_LanguageMode0, false);
      ToolErrorReporter toolErrorReporter0 = new ToolErrorReporter(false);
      // Undeclared exception!
      try { 
        IRFactory.transformTree(astRoot0, (StaticSourceFile) null, "unsupported language extension: for each", config0, toolErrorReporter0);
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
      SimpleSourceFile simpleSourceFile0 = new SimpleSourceFile("$1ds%oO/9%SsoKrS", true);
      LinkedHashSet<String> linkedHashSet0 = new LinkedHashSet<String>();
      ToolErrorReporter toolErrorReporter0 = new ToolErrorReporter(true);
      NewExpression newExpression0 = new NewExpression(4);
      astRoot0.addChildrenToBack(newExpression0);
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5_STRICT;
      Config config0 = new Config(linkedHashSet0, linkedHashSet0, true, config_LanguageMode0, false);
      // Undeclared exception!
      try { 
        IRFactory.transformTree(astRoot0, simpleSourceFile0, "BUDI'No-A&Afj9jp{z", config0, toolErrorReporter0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
      }
  }

  @Test(timeout = 4000)
  public void test38()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      TreeSet<String> treeSet0 = new TreeSet<String>();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5_STRICT;
      Config config0 = new Config(treeSet0, treeSet0, false, config_LanguageMode0, false);
      FunctionCall functionCall0 = new FunctionCall(153, 2);
      astRoot0.addChildrenToBack(functionCall0);
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
  public void test39()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      TreeSet<String> treeSet0 = new TreeSet<String>();
      ExpressionStatement expressionStatement0 = new ExpressionStatement();
      expressionStatement0.setHasResult();
      astRoot0.addChildrenToBack(expressionStatement0);
      ToolErrorReporter toolErrorReporter0 = new ToolErrorReporter(false);
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5_STRICT;
      Config config0 = new Config(treeSet0, treeSet0, false, config_LanguageMode0, false);
      // Undeclared exception!
      try { 
        IRFactory.transformTree(astRoot0, (StaticSourceFile) null, "", config0, toolErrorReporter0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
      }
  }
}
