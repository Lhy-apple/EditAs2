/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 05:05:45 GMT 2023
 */

package com.google.javascript.jscomp.parsing;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.google.common.collect.ImmutableList;
import com.google.javascript.jscomp.parsing.Config;
import com.google.javascript.jscomp.parsing.IRFactory;
import com.google.javascript.rhino.head.Context;
import com.google.javascript.rhino.head.ErrorReporter;
import com.google.javascript.rhino.head.Node;
import com.google.javascript.rhino.head.Token;
import com.google.javascript.rhino.head.ast.AstRoot;
import com.google.javascript.rhino.head.ast.Block;
import com.google.javascript.rhino.head.ast.BreakStatement;
import com.google.javascript.rhino.head.ast.Comment;
import com.google.javascript.rhino.head.ast.ConditionalExpression;
import com.google.javascript.rhino.head.ast.ContinueStatement;
import com.google.javascript.rhino.head.ast.DoLoop;
import com.google.javascript.rhino.head.ast.ElementGet;
import com.google.javascript.rhino.head.ast.EmptyStatement;
import com.google.javascript.rhino.head.ast.ErrorCollector;
import com.google.javascript.rhino.head.ast.ExpressionStatement;
import com.google.javascript.rhino.head.ast.ForLoop;
import com.google.javascript.rhino.head.ast.FunctionNode;
import com.google.javascript.rhino.head.ast.GeneratorExpressionLoop;
import com.google.javascript.rhino.head.ast.Label;
import com.google.javascript.rhino.head.ast.LabeledStatement;
import com.google.javascript.rhino.head.ast.Name;
import com.google.javascript.rhino.head.ast.ObjectProperty;
import com.google.javascript.rhino.head.ast.ParenthesizedExpression;
import com.google.javascript.rhino.head.ast.PropertyGet;
import com.google.javascript.rhino.head.ast.RegExpLiteral;
import com.google.javascript.rhino.head.ast.ReturnStatement;
import com.google.javascript.rhino.head.ast.Scope;
import com.google.javascript.rhino.head.ast.ThrowStatement;
import com.google.javascript.rhino.head.ast.UnaryExpression;
import com.google.javascript.rhino.head.ast.VariableDeclaration;
import com.google.javascript.rhino.head.ast.WhileLoop;
import com.google.javascript.rhino.head.ast.WithStatement;
import com.google.javascript.rhino.head.ast.XmlElemRef;
import com.google.javascript.rhino.head.tools.ToolErrorReporter;
import com.google.javascript.rhino.jstype.SimpleSourceFile;
import com.google.javascript.rhino.jstype.StaticSourceFile;
import java.util.HashSet;
import java.util.LinkedHashSet;
import java.util.TreeSet;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class IRFactory_ESTest extends IRFactory_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      ConditionalExpression conditionalExpression0 = new ConditionalExpression(0, 1);
      astRoot0.addChildToFront(conditionalExpression0);
      LinkedHashSet<String> linkedHashSet0 = new LinkedHashSet<String>();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5_STRICT;
      Config config0 = new Config(linkedHashSet0, linkedHashSet0, false, config_LanguageMode0, false);
      ToolErrorReporter toolErrorReporter0 = new ToolErrorReporter(true);
      // Undeclared exception!
      try { 
        IRFactory.transformTree(astRoot0, (StaticSourceFile) null, (String) null, config0, toolErrorReporter0);
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
      ParenthesizedExpression parenthesizedExpression0 = new ParenthesizedExpression((-875), 0);
      astRoot0.addChildToFront(parenthesizedExpression0);
      LinkedHashSet<String> linkedHashSet0 = new LinkedHashSet<String>();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT3;
      Config config0 = new Config(linkedHashSet0, linkedHashSet0, true, config_LanguageMode0, true);
      ErrorCollector errorCollector0 = new ErrorCollector();
      // Undeclared exception!
      try { 
        IRFactory.transformTree(astRoot0, (StaticSourceFile) null, "`QoS5+>Zu,8zpI", config0, errorCollector0);
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
      DoLoop doLoop0 = new DoLoop(11);
      astRoot0.addChildToFront(doLoop0);
      SimpleSourceFile simpleSourceFile0 = new SimpleSourceFile("4*;EhR?~lV>Fi", true);
      LinkedHashSet<String> linkedHashSet0 = new LinkedHashSet<String>();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5_STRICT;
      Config config0 = new Config(linkedHashSet0, linkedHashSet0, true, config_LanguageMode0, true);
      ErrorCollector errorCollector0 = new ErrorCollector();
      // Undeclared exception!
      try { 
        IRFactory.transformTree(astRoot0, simpleSourceFile0, "4*;EhR?~lV>Fi", config0, errorCollector0);
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
      LinkedHashSet<String> linkedHashSet0 = new LinkedHashSet<String>(23);
      ElementGet elementGet0 = new ElementGet((-2874));
      astRoot0.addChildToFront(elementGet0);
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5;
      Config config0 = new Config(linkedHashSet0, linkedHashSet0, true, config_LanguageMode0, false);
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
  public void test04()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      PropertyGet propertyGet0 = new PropertyGet(1, 18);
      astRoot0.addChildToFront(propertyGet0);
      LinkedHashSet<String> linkedHashSet0 = new LinkedHashSet<String>();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT3;
      Config config0 = new Config(linkedHashSet0, linkedHashSet0, false, config_LanguageMode0, false);
      propertyGet0.setOperator(14);
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
  public void test05()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      WithStatement withStatement0 = new WithStatement();
      astRoot0.addChildToFront(withStatement0);
      LinkedHashSet<String> linkedHashSet0 = new LinkedHashSet<String>();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5_STRICT;
      Config config0 = new Config(linkedHashSet0, linkedHashSet0, false, config_LanguageMode0, false);
      ErrorCollector errorCollector0 = new ErrorCollector();
      // Undeclared exception!
      try { 
        IRFactory.transformTree(astRoot0, (StaticSourceFile) null, "const", config0, errorCollector0);
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
      ForLoop forLoop0 = new ForLoop();
      astRoot0.addChildToFront(forLoop0);
      LinkedHashSet<String> linkedHashSet0 = new LinkedHashSet<String>(2);
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5;
      Config config0 = new Config(linkedHashSet0, linkedHashSet0, false, config_LanguageMode0, true);
      ErrorCollector errorCollector0 = new ErrorCollector();
      // Undeclared exception!
      try { 
        IRFactory.transformTree(astRoot0, (StaticSourceFile) null, "N5Ny2b3D^ly6I", config0, errorCollector0);
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
      LabeledStatement labeledStatement0 = new LabeledStatement(11);
      Label label0 = new Label(6, 4);
      ImmutableList<Label> immutableList0 = ImmutableList.of(label0, label0, label0, label0);
      labeledStatement0.setLabels(immutableList0);
      astRoot0.addChildToFront(labeledStatement0);
      LinkedHashSet<String> linkedHashSet0 = new LinkedHashSet<String>();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5;
      Config config0 = new Config(linkedHashSet0, linkedHashSet0, false, config_LanguageMode0, false);
      ErrorCollector errorCollector0 = new ErrorCollector();
      // Undeclared exception!
      try { 
        IRFactory.transformTree(astRoot0, (StaticSourceFile) null, "->Yaz'KK", config0, errorCollector0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // StringNode: str is null
         //
         verifyException("com.google.javascript.rhino.Node$StringNode", e);
      }
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      Node node0 = Node.newNumber(22);
      astRoot0.addChildToFront(node0);
      LinkedHashSet<String> linkedHashSet0 = new LinkedHashSet<String>(26);
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5;
      Config config0 = new Config(linkedHashSet0, linkedHashSet0, true, config_LanguageMode0, true);
      ErrorCollector errorCollector0 = new ErrorCollector();
      com.google.javascript.rhino.Node node1 = IRFactory.transformTree(astRoot0, (StaticSourceFile) null, "e c", config0, errorCollector0);
      assertEquals(1, node1.getLength());
      assertTrue(node1.isScript());
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      Name name0 = new Name(20, (-28), "");
      ExpressionStatement expressionStatement0 = new ExpressionStatement(name0, true);
      astRoot0.addChildToFront(expressionStatement0);
      SimpleSourceFile simpleSourceFile0 = new SimpleSourceFile("),", true);
      LinkedHashSet<String> linkedHashSet0 = new LinkedHashSet<String>();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5_STRICT;
      Config config0 = new Config(linkedHashSet0, linkedHashSet0, false, config_LanguageMode0, true);
      ErrorCollector errorCollector0 = new ErrorCollector();
      com.google.javascript.rhino.Node node0 = IRFactory.transformTree(astRoot0, simpleSourceFile0, "),", config0, errorCollector0);
      assertTrue(node0.isFromExterns());
      assertEquals(0, node0.getLength());
      assertTrue(node0.isScript());
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      LinkedHashSet<String> linkedHashSet0 = new LinkedHashSet<String>(23);
      EmptyStatement emptyStatement0 = new EmptyStatement();
      astRoot0.addChildToFront(emptyStatement0);
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5;
      Config config0 = new Config(linkedHashSet0, linkedHashSet0, true, config_LanguageMode0, true);
      ErrorCollector errorCollector0 = new ErrorCollector();
      SimpleSourceFile simpleSourceFile0 = new SimpleSourceFile("javax", false);
      com.google.javascript.rhino.Node node0 = IRFactory.transformTree(astRoot0, simpleSourceFile0, "javax", config0, errorCollector0);
      assertTrue(node0.isScript());
      assertEquals(1, node0.getLength());
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      Scope scope0 = new Scope(3);
      astRoot0.addChildToFront(scope0);
      SimpleSourceFile simpleSourceFile0 = new SimpleSourceFile("", false);
      LinkedHashSet<String> linkedHashSet0 = new LinkedHashSet<String>();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5_STRICT;
      Config config0 = new Config(linkedHashSet0, linkedHashSet0, false, config_LanguageMode0, false);
      ErrorCollector errorCollector0 = new ErrorCollector();
      com.google.javascript.rhino.Node node0 = IRFactory.transformTree(astRoot0, simpleSourceFile0, "", config0, errorCollector0);
      assertEquals(0, node0.getLength());
      assertEquals(132, node0.getType());
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      WhileLoop whileLoop0 = new WhileLoop();
      astRoot0.addChildToFront(whileLoop0);
      LinkedHashSet<String> linkedHashSet0 = new LinkedHashSet<String>();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5_STRICT;
      Config config0 = new Config(linkedHashSet0, linkedHashSet0, true, config_LanguageMode0, false);
      Context.getCurrentContext();
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      Block block0 = new Block(24, 11);
      astRoot0.addChildToFront(block0);
      SimpleSourceFile simpleSourceFile0 = new SimpleSourceFile("com.google.common.collect.Sets$1", false);
      LinkedHashSet<String> linkedHashSet0 = new LinkedHashSet<String>();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5;
      Config config0 = new Config(linkedHashSet0, linkedHashSet0, false, config_LanguageMode0, false);
      Context.getCurrentContext();
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      ObjectProperty objectProperty0 = new ObjectProperty();
      ExpressionStatement expressionStatement0 = new ExpressionStatement(objectProperty0, false);
      astRoot0.addChildToFront(expressionStatement0);
      HashSet<String> hashSet0 = new HashSet<String>();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT3;
      Config config0 = new Config(hashSet0, hashSet0, true, config_LanguageMode0, false);
      ToolErrorReporter toolErrorReporter0 = new ToolErrorReporter(false);
      // Undeclared exception!
      try { 
        IRFactory.transformTree(astRoot0, (StaticSourceFile) null, "LTy;g<y~1oN|n^O", config0, toolErrorReporter0);
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // 103
         //
         verifyException("com.google.javascript.jscomp.parsing.IRFactory", e);
      }
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      XmlElemRef xmlElemRef0 = new XmlElemRef();
      astRoot0.addChildToFront(xmlElemRef0);
      LinkedHashSet<String> linkedHashSet0 = new LinkedHashSet<String>(1);
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5_STRICT;
      Config config0 = new Config(linkedHashSet0, linkedHashSet0, true, config_LanguageMode0, true);
      ErrorCollector errorCollector0 = new ErrorCollector();
      // Undeclared exception!
      try { 
        IRFactory.transformTree(astRoot0, (StaticSourceFile) null, "private", config0, errorCollector0);
        fail("Expecting exception: UnsupportedOperationException");
      
      } catch(UnsupportedOperationException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.rhino.head.ast.ErrorCollector", e);
      }
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      ThrowStatement throwStatement0 = new ThrowStatement(20, 3244);
      astRoot0.addChildToFront(throwStatement0);
      SimpleSourceFile simpleSourceFile0 = new SimpleSourceFile("", false);
      LinkedHashSet<String> linkedHashSet0 = new LinkedHashSet<String>(9);
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5_STRICT;
      Config config0 = new Config(linkedHashSet0, linkedHashSet0, false, config_LanguageMode0, false);
      Context.getCurrentContext();
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      FunctionNode functionNode0 = new FunctionNode();
      astRoot0.addChildToFront(functionNode0);
      HashSet<String> hashSet0 = new HashSet<String>();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5;
      Config config0 = new Config(hashSet0, hashSet0, false, config_LanguageMode0, false);
      ToolErrorReporter toolErrorReporter0 = new ToolErrorReporter(false);
      com.google.javascript.rhino.Node node0 = IRFactory.transformTree(astRoot0, (StaticSourceFile) null, "", config0, toolErrorReporter0);
      assertEquals(0, node0.getLength());
      assertEquals(132, node0.getType());
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      Token.CommentType token_CommentType0 = Token.CommentType.HTML;
      Comment comment0 = new Comment(8, 25, token_CommentType0, "javax");
      astRoot0.setJsDocNode(comment0);
      LinkedHashSet<String> linkedHashSet0 = new LinkedHashSet<String>(4);
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT3;
      Config config0 = new Config(linkedHashSet0, linkedHashSet0, true, config_LanguageMode0, true);
      ToolErrorReporter toolErrorReporter0 = new ToolErrorReporter(true);
      com.google.javascript.rhino.Node node0 = IRFactory.transformTree(astRoot0, (StaticSourceFile) null, "javax", config0, toolErrorReporter0);
      assertTrue(node0.isScript());
      assertEquals(1, node0.getLength());
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      Token.CommentType token_CommentType0 = Token.CommentType.HTML;
      Comment comment0 = new Comment(8, 25, token_CommentType0, "javax");
      astRoot0.setJsDocNode(comment0);
      LinkedHashSet<String> linkedHashSet0 = new LinkedHashSet<String>(4);
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT3;
      Config config0 = new Config(linkedHashSet0, linkedHashSet0, true, config_LanguageMode0, true);
      ErrorCollector errorCollector0 = new ErrorCollector();
      // Undeclared exception!
      try { 
        IRFactory.transformTree(astRoot0, (StaticSourceFile) null, "\n * @", config0, errorCollector0);
        fail("Expecting exception: UnsupportedOperationException");
      
      } catch(UnsupportedOperationException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.rhino.head.ast.ErrorCollector", e);
      }
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      LinkedHashSet<String> linkedHashSet0 = new LinkedHashSet<String>(23);
      BreakStatement breakStatement0 = new BreakStatement(9);
      astRoot0.addChildToFront(breakStatement0);
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5;
      Config config0 = new Config(linkedHashSet0, linkedHashSet0, true, config_LanguageMode0, true);
      ErrorCollector errorCollector0 = new ErrorCollector();
      com.google.javascript.rhino.Node node0 = IRFactory.transformTree(astRoot0, (StaticSourceFile) null, "R_,\"BGiLflGW", config0, errorCollector0);
      assertTrue(node0.isScript());
      assertEquals(1, node0.getLength());
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      ContinueStatement continueStatement0 = new ContinueStatement((Name) null);
      astRoot0.addChildToFront(continueStatement0);
      LinkedHashSet<String> linkedHashSet0 = new LinkedHashSet<String>();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5_STRICT;
      Config config0 = new Config(linkedHashSet0, linkedHashSet0, false, config_LanguageMode0, false);
      ErrorCollector errorCollector0 = new ErrorCollector();
      // Undeclared exception!
      try { 
        IRFactory.transformTree(astRoot0, (StaticSourceFile) null, (String) null, config0, errorCollector0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.parsing.IRFactory", e);
      }
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      Name name0 = new Name(7, 114, "epxe,*]j)Pu0g<!{OUa");
      ContinueStatement continueStatement0 = new ContinueStatement(name0);
      astRoot0.addChildToFront(continueStatement0);
      TreeSet<String> treeSet0 = new TreeSet<String>();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5_STRICT;
      Config config0 = new Config(treeSet0, treeSet0, false, config_LanguageMode0, false);
      ErrorCollector errorCollector0 = new ErrorCollector();
      com.google.javascript.rhino.Node node0 = IRFactory.transformTree(astRoot0, (StaticSourceFile) null, "epxe,*]j)Pu0g<!{OUa", config0, errorCollector0);
      assertEquals(0, node0.getLength());
      assertTrue(node0.isScript());
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      GeneratorExpressionLoop generatorExpressionLoop0 = new GeneratorExpressionLoop(1);
      astRoot0.addChildToFront(generatorExpressionLoop0);
      SimpleSourceFile simpleSourceFile0 = new SimpleSourceFile("T", false);
      LinkedHashSet<String> linkedHashSet0 = new LinkedHashSet<String>();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5_STRICT;
      Config config0 = new Config(linkedHashSet0, linkedHashSet0, false, config_LanguageMode0, false);
      ErrorCollector errorCollector0 = new ErrorCollector();
      // Undeclared exception!
      try { 
        IRFactory.transformTree(astRoot0, simpleSourceFile0, "T", config0, errorCollector0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
      }
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      FunctionNode functionNode0 = new FunctionNode();
      Name name0 = new Name(1);
      functionNode0.setFunctionName(name0);
      astRoot0.addChildToFront(functionNode0);
      HashSet<String> hashSet0 = new HashSet<String>();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5_STRICT;
      Config config0 = new Config(hashSet0, hashSet0, true, config_LanguageMode0, true);
      ErrorCollector errorCollector0 = new ErrorCollector();
      // Undeclared exception!
      try { 
        IRFactory.transformTree(astRoot0, (StaticSourceFile) null, "Non-JSDoc comment has annotations. Did you mean to start it with '/**'?", config0, errorCollector0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // StringNode: str is null
         //
         verifyException("com.google.javascript.rhino.Node$StringNode", e);
      }
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      LabeledStatement labeledStatement0 = new LabeledStatement(11, 17);
      astRoot0.addChildToFront(labeledStatement0);
      SimpleSourceFile simpleSourceFile0 = new SimpleSourceFile("),", true);
      LinkedHashSet<String> linkedHashSet0 = new LinkedHashSet<String>();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5_STRICT;
      Config config0 = new Config(linkedHashSet0, linkedHashSet0, true, config_LanguageMode0, true);
      ErrorCollector errorCollector0 = new ErrorCollector();
      // Undeclared exception!
      try { 
        IRFactory.transformTree(astRoot0, simpleSourceFile0, "),", config0, errorCollector0);
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
      Name name0 = new Name(25);
      astRoot0.addChildToFront(name0);
      LinkedHashSet<String> linkedHashSet0 = new LinkedHashSet<String>(23);
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5;
      name0.setIdentifier("super");
      Config config0 = new Config(linkedHashSet0, linkedHashSet0, true, config_LanguageMode0, true);
      ErrorCollector errorCollector0 = new ErrorCollector();
      // Undeclared exception!
      try { 
        IRFactory.transformTree(astRoot0, (StaticSourceFile) null, "`qD@[O3Y", config0, errorCollector0);
        fail("Expecting exception: UnsupportedOperationException");
      
      } catch(UnsupportedOperationException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.rhino.head.ast.ErrorCollector", e);
      }
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      Name name0 = new Name(25);
      astRoot0.addChildToFront(name0);
      LinkedHashSet<String> linkedHashSet0 = new LinkedHashSet<String>(18);
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT3;
      Config config0 = new Config(linkedHashSet0, linkedHashSet0, false, config_LanguageMode0, false);
      ErrorCollector errorCollector0 = new ErrorCollector();
      // Undeclared exception!
      try { 
        IRFactory.transformTree(astRoot0, (StaticSourceFile) null, "N5Ny2b3D^ly6I", config0, errorCollector0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.rhino.TokenStream", e);
      }
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      RegExpLiteral regExpLiteral0 = new RegExpLiteral(23);
      regExpLiteral0.setValue("");
      astRoot0.addChildToFront(regExpLiteral0);
      LinkedHashSet<String> linkedHashSet0 = new LinkedHashSet<String>();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5;
      Config config0 = new Config(linkedHashSet0, linkedHashSet0, false, config_LanguageMode0, false);
      ErrorCollector errorCollector0 = new ErrorCollector();
      com.google.javascript.rhino.Node node0 = IRFactory.transformTree(astRoot0, (StaticSourceFile) null, "d", config0, errorCollector0);
      assertEquals(0, node0.getLength());
      assertEquals(132, node0.getType());
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      RegExpLiteral regExpLiteral0 = new RegExpLiteral(16);
      regExpLiteral0.setFlags("");
      regExpLiteral0.setValue("let");
      astRoot0.addChildToFront(regExpLiteral0);
      LinkedHashSet<String> linkedHashSet0 = new LinkedHashSet<String>();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5;
      Config config0 = new Config(linkedHashSet0, linkedHashSet0, true, config_LanguageMode0, false);
      ErrorCollector errorCollector0 = new ErrorCollector();
      com.google.javascript.rhino.Node node0 = IRFactory.transformTree(astRoot0, (StaticSourceFile) null, "let", config0, errorCollector0);
      assertTrue(node0.isScript());
      assertEquals(1, node0.getLength());
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      RegExpLiteral regExpLiteral0 = new RegExpLiteral(23);
      regExpLiteral0.setFlags("Vg^]KqByf3CT|8,F");
      regExpLiteral0.setValue("b");
      astRoot0.addChildToFront(regExpLiteral0);
      LinkedHashSet<String> linkedHashSet0 = new LinkedHashSet<String>();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5;
      Config config0 = new Config(linkedHashSet0, linkedHashSet0, true, config_LanguageMode0, true);
      ErrorCollector errorCollector0 = new ErrorCollector();
      com.google.javascript.rhino.Node node0 = IRFactory.transformTree(astRoot0, (StaticSourceFile) null, "'.(xl,7H7.+{HgmJ", config0, errorCollector0);
      assertTrue(node0.isScript());
      assertEquals(1, node0.getLength());
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      ReturnStatement returnStatement0 = new ReturnStatement();
      astRoot0.addChildToFront(returnStatement0);
      LinkedHashSet<String> linkedHashSet0 = new LinkedHashSet<String>(25);
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT3;
      Config config0 = new Config(linkedHashSet0, linkedHashSet0, true, config_LanguageMode0, true);
      ErrorCollector errorCollector0 = new ErrorCollector();
      com.google.javascript.rhino.Node node0 = IRFactory.transformTree(astRoot0, (StaticSourceFile) null, "-<fUZ", config0, errorCollector0);
      assertEquals(1, node0.getLength());
      assertEquals(132, node0.getType());
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      ReturnStatement returnStatement0 = new ReturnStatement(119, 24, astRoot0);
      astRoot0.addChildToFront(returnStatement0);
      LinkedHashSet<String> linkedHashSet0 = new LinkedHashSet<String>(25);
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT3;
      Config config0 = new Config(linkedHashSet0, linkedHashSet0, true, config_LanguageMode0, true);
      ErrorCollector errorCollector0 = new ErrorCollector();
      // Undeclared exception!
      try { 
        IRFactory.transformTree(astRoot0, (StaticSourceFile) null, "-<fUZ", config0, errorCollector0);
        fail("Expecting exception: StackOverflowError");
      
      } catch(StackOverflowError e) {
         //
         // no message in exception (getMessage() returned null)
         //
      }
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      VariableDeclaration variableDeclaration0 = new VariableDeclaration(7);
      astRoot0.addChildToFront(variableDeclaration0);
      LinkedHashSet<String> linkedHashSet0 = new LinkedHashSet<String>();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5_STRICT;
      Config config0 = new Config(linkedHashSet0, linkedHashSet0, true, config_LanguageMode0, true);
      ErrorCollector errorCollector0 = new ErrorCollector();
      // Undeclared exception!
      try { 
        IRFactory.transformTree(astRoot0, (StaticSourceFile) null, (String) null, config0, errorCollector0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.parsing.IRFactory", e);
      }
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      VariableDeclaration variableDeclaration0 = new VariableDeclaration(7);
      astRoot0.addChildToFront(variableDeclaration0);
      LinkedHashSet<String> linkedHashSet0 = new LinkedHashSet<String>();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5_STRICT;
      Config config0 = new Config(linkedHashSet0, linkedHashSet0, false, config_LanguageMode0, false);
      ErrorCollector errorCollector0 = new ErrorCollector();
      com.google.javascript.rhino.Node node0 = IRFactory.transformTree(astRoot0, (StaticSourceFile) null, "super", config0, errorCollector0);
      assertTrue(node0.isScript());
      assertEquals(0, node0.getLength());
  }

  @Test(timeout = 4000)
  public void test35()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      PropertyGet propertyGet0 = new PropertyGet();
      astRoot0.addChildToFront(propertyGet0);
      LinkedHashSet<String> linkedHashSet0 = new LinkedHashSet<String>();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5_STRICT;
      Config config0 = new Config(linkedHashSet0, linkedHashSet0, true, config_LanguageMode0, true);
      propertyGet0.setOperator(16);
      ErrorCollector errorCollector0 = new ErrorCollector();
      // Undeclared exception!
      try { 
        IRFactory.transformTree(astRoot0, (StaticSourceFile) null, "applet", config0, errorCollector0);
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
      UnaryExpression unaryExpression0 = new UnaryExpression(26, 8, astRoot0, true);
      astRoot0.addChildToFront(unaryExpression0);
      LinkedHashSet<String> linkedHashSet0 = new LinkedHashSet<String>();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT3;
      Config config0 = new Config(linkedHashSet0, linkedHashSet0, true, config_LanguageMode0, true);
      ErrorCollector errorCollector0 = new ErrorCollector();
      // Undeclared exception!
      try { 
        IRFactory.transformTree(astRoot0, (StaticSourceFile) null, "0!?_S{0E2&:-IE7<1$T", config0, errorCollector0);
        fail("Expecting exception: StackOverflowError");
      
      } catch(StackOverflowError e) {
         //
         // no message in exception (getMessage() returned null)
         //
      }
  }
}