/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 02:26:26 GMT 2023
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
import com.google.javascript.rhino.head.ast.ArrayLiteral;
import com.google.javascript.rhino.head.ast.Assignment;
import com.google.javascript.rhino.head.ast.AstRoot;
import com.google.javascript.rhino.head.ast.BreakStatement;
import com.google.javascript.rhino.head.ast.Comment;
import com.google.javascript.rhino.head.ast.ConditionalExpression;
import com.google.javascript.rhino.head.ast.ContinueStatement;
import com.google.javascript.rhino.head.ast.DoLoop;
import com.google.javascript.rhino.head.ast.ElementGet;
import com.google.javascript.rhino.head.ast.EmptyExpression;
import com.google.javascript.rhino.head.ast.EmptyStatement;
import com.google.javascript.rhino.head.ast.ErrorCollector;
import com.google.javascript.rhino.head.ast.ExpressionStatement;
import com.google.javascript.rhino.head.ast.FunctionCall;
import com.google.javascript.rhino.head.ast.FunctionNode;
import com.google.javascript.rhino.head.ast.GeneratorExpressionLoop;
import com.google.javascript.rhino.head.ast.IfStatement;
import com.google.javascript.rhino.head.ast.Label;
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
import com.google.javascript.rhino.head.ast.StringLiteral;
import com.google.javascript.rhino.head.ast.SwitchCase;
import com.google.javascript.rhino.head.ast.ThrowStatement;
import com.google.javascript.rhino.head.ast.VariableDeclaration;
import com.google.javascript.rhino.head.ast.WhileLoop;
import com.google.javascript.rhino.head.ast.WithStatement;
import com.google.javascript.rhino.head.tools.ToolErrorReporter;
import com.google.javascript.rhino.jstype.SimpleSourceFile;
import com.google.javascript.rhino.jstype.StaticSourceFile;
import java.util.LinkedHashSet;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class IRFactory_ESTest extends IRFactory_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      LinkedHashSet<String> linkedHashSet0 = new LinkedHashSet<String>();
      ConditionalExpression conditionalExpression0 = new ConditionalExpression();
      astRoot0.addChildToBack(conditionalExpression0);
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5;
      Config config0 = new Config(linkedHashSet0, linkedHashSet0, false, config_LanguageMode0, false);
      ErrorCollector errorCollector0 = new ErrorCollector();
      // Undeclared exception!
      try { 
        IRFactory.transformTree(astRoot0, (StaticSourceFile) null, "getters may not have parameters", config0, errorCollector0);
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
      LinkedHashSet<String> linkedHashSet0 = new LinkedHashSet<String>();
      ParenthesizedExpression parenthesizedExpression0 = new ParenthesizedExpression(4, 1, astRoot0);
      astRoot0.addChildToBack(parenthesizedExpression0);
      ToolErrorReporter toolErrorReporter0 = new ToolErrorReporter(true);
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5_STRICT;
      Config config0 = new Config(linkedHashSet0, linkedHashSet0, true, config_LanguageMode0, true);
      // Undeclared exception!
      try { 
        IRFactory.transformTree(astRoot0, (StaticSourceFile) null, "language version", config0, toolErrorReporter0);
        fail("Expecting exception: StackOverflowError");
      
      } catch(StackOverflowError e) {
         //
         // no message in exception (getMessage() returned null)
         //
      }
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      LinkedHashSet<String> linkedHashSet0 = new LinkedHashSet<String>();
      DoLoop doLoop0 = new DoLoop();
      astRoot0.addChildToBack(doLoop0);
      ToolErrorReporter toolErrorReporter0 = new ToolErrorReporter(false);
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5_STRICT;
      Config config0 = new Config(linkedHashSet0, linkedHashSet0, false, config_LanguageMode0, false);
      // Undeclared exception!
      try { 
        IRFactory.transformTree(astRoot0, (StaticSourceFile) null, "error reporter", config0, toolErrorReporter0);
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
      EmptyExpression emptyExpression0 = new EmptyExpression();
      astRoot0.addChildToBack(emptyExpression0);
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5;
      Config config0 = new Config(linkedHashSet0, linkedHashSet0, false, config_LanguageMode0, false);
      ErrorCollector errorCollector0 = new ErrorCollector();
      Node node0 = IRFactory.transformTree(astRoot0, (StaticSourceFile) null, "error reporter", config0, errorCollector0);
      assertEquals(1, node0.getChildCount());
      assertTrue(node0.isScript());
      assertEquals(0, node0.getLength());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      LinkedHashSet<String> linkedHashSet0 = new LinkedHashSet<String>();
      ElementGet elementGet0 = new ElementGet(13);
      astRoot0.addChildToBack(elementGet0);
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT3;
      Config config0 = new Config(linkedHashSet0, linkedHashSet0, true, config_LanguageMode0, false);
      ErrorCollector errorCollector0 = new ErrorCollector();
      // Undeclared exception!
      try { 
        IRFactory.transformTree(astRoot0, (StaticSourceFile) null, "e@VjH<u", config0, errorCollector0);
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
      ObjectProperty objectProperty0 = new ObjectProperty(2190, 2492);
      astRoot0.addChildToBack(objectProperty0);
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5_STRICT;
      Config config0 = new Config(linkedHashSet0, linkedHashSet0, true, config_LanguageMode0, true);
      ToolErrorReporter toolErrorReporter0 = new ToolErrorReporter(true);
      // Undeclared exception!
      try { 
        IRFactory.transformTree(astRoot0, (StaticSourceFile) null, "language version", config0, toolErrorReporter0);
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
      LinkedHashSet<String> linkedHashSet0 = new LinkedHashSet<String>();
      SwitchCase switchCase0 = new SwitchCase(8);
      WithStatement withStatement0 = new WithStatement(24, (-2452));
      switchCase0.addStatement(withStatement0);
      astRoot0.addChildToBack(switchCase0);
      ToolErrorReporter toolErrorReporter0 = new ToolErrorReporter(true);
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5_STRICT;
      Config config0 = new Config(linkedHashSet0, linkedHashSet0, true, config_LanguageMode0, false);
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
  public void test07()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      LinkedHashSet<String> linkedHashSet0 = new LinkedHashSet<String>();
      Label label0 = new Label();
      astRoot0.addChildToBack(label0);
      Context.getCurrentContext();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5;
      Config config0 = new Config(linkedHashSet0, linkedHashSet0, false, config_LanguageMode0, true);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      LinkedHashSet<String> linkedHashSet0 = new LinkedHashSet<String>();
      NumberLiteral numberLiteral0 = new NumberLiteral();
      astRoot0.addChildToBack(numberLiteral0);
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT3;
      Config config0 = new Config(linkedHashSet0, linkedHashSet0, true, config_LanguageMode0, true);
      Context.getCurrentContext();
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      LinkedHashSet<String> linkedHashSet0 = new LinkedHashSet<String>();
      VariableDeclaration variableDeclaration0 = new VariableDeclaration(6);
      ExpressionStatement expressionStatement0 = new ExpressionStatement(variableDeclaration0, true);
      astRoot0.addChildToBack(expressionStatement0);
      ToolErrorReporter toolErrorReporter0 = new ToolErrorReporter(true);
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5;
      Config config0 = new Config(linkedHashSet0, linkedHashSet0, true, config_LanguageMode0, true);
      Node node0 = IRFactory.transformTree(astRoot0, (StaticSourceFile) null, "error reporter", config0, toolErrorReporter0);
      assertEquals(1, node0.getLength());
      assertEquals(132, node0.getType());
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      LinkedHashSet<String> linkedHashSet0 = new LinkedHashSet<String>();
      EmptyStatement emptyStatement0 = new EmptyStatement(2);
      astRoot0.addChildToBack(emptyStatement0);
      ToolErrorReporter toolErrorReporter0 = new ToolErrorReporter(true);
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5_STRICT;
      Config config0 = new Config(linkedHashSet0, linkedHashSet0, true, config_LanguageMode0, true);
      Node node0 = IRFactory.transformTree(astRoot0, (StaticSourceFile) null, "error reporter", config0, toolErrorReporter0);
      assertEquals(1, node0.getLength());
      assertEquals(132, node0.getType());
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      LinkedHashSet<String> linkedHashSet0 = new LinkedHashSet<String>();
      Scope scope0 = new Scope(13, 2);
      astRoot0.addChildToBack(scope0);
      ToolErrorReporter toolErrorReporter0 = new ToolErrorReporter(false);
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5_STRICT;
      Config config0 = new Config(linkedHashSet0, linkedHashSet0, false, config_LanguageMode0, false);
      Node node0 = IRFactory.transformTree(astRoot0, (StaticSourceFile) null, "Upd+[Pzj%Tw)R6X4*J", config0, toolErrorReporter0);
      assertEquals(0, node0.getLength());
      assertEquals(132, node0.getType());
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      LinkedHashSet<String> linkedHashSet0 = new LinkedHashSet<String>();
      WhileLoop whileLoop0 = new WhileLoop(1540, 21);
      astRoot0.addChildToBack(whileLoop0);
      ToolErrorReporter toolErrorReporter0 = new ToolErrorReporter(true);
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5_STRICT;
      Config config0 = new Config(linkedHashSet0, linkedHashSet0, true, config_LanguageMode0, true);
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
  public void test13()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      LinkedHashSet<String> linkedHashSet0 = new LinkedHashSet<String>(18);
      IfStatement ifStatement0 = new IfStatement(23, 5);
      Name name0 = new Name(25, "jf(0#");
      PropertyGet propertyGet0 = new PropertyGet(18, 1582, ifStatement0, name0);
      propertyGet0.setTarget(name0);
      astRoot0.addChildToBack(propertyGet0);
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5;
      Config config0 = new Config(linkedHashSet0, linkedHashSet0, true, config_LanguageMode0, true);
      ErrorCollector errorCollector0 = new ErrorCollector();
      Node node0 = IRFactory.transformTree(astRoot0, (StaticSourceFile) null, ")WEO_Ve&q", config0, errorCollector0);
      assertEquals(1, node0.getLength());
      assertEquals(132, node0.getType());
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      LinkedHashSet<String> linkedHashSet0 = new LinkedHashSet<String>();
      Assignment assignment0 = new Assignment();
      astRoot0.addChildToBack(assignment0);
      ToolErrorReporter toolErrorReporter0 = new ToolErrorReporter(false);
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5_STRICT;
      Config config0 = new Config(linkedHashSet0, linkedHashSet0, true, config_LanguageMode0, true);
      Node node0 = IRFactory.transformTree(astRoot0, (StaticSourceFile) null, "language version", config0, toolErrorReporter0);
      assertEquals(1, node0.getLength());
      assertTrue(node0.isScript());
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      LinkedHashSet<String> linkedHashSet0 = new LinkedHashSet<String>();
      ThrowStatement throwStatement0 = new ThrowStatement(astRoot0);
      astRoot0.addChildToBack(throwStatement0);
      ToolErrorReporter toolErrorReporter0 = new ToolErrorReporter(false);
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT3;
      Config config0 = new Config(linkedHashSet0, linkedHashSet0, true, config_LanguageMode0, false);
      // Undeclared exception!
      try { 
        IRFactory.transformTree(astRoot0, (StaticSourceFile) null, "Not declared as a constructor", config0, toolErrorReporter0);
        fail("Expecting exception: StackOverflowError");
      
      } catch(StackOverflowError e) {
         //
         // no message in exception (getMessage() returned null)
         //
      }
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      LinkedHashSet<String> linkedHashSet0 = new LinkedHashSet<String>();
      ArrayLiteral arrayLiteral0 = new ArrayLiteral(12);
      NewExpression newExpression0 = new NewExpression();
      newExpression0.setTarget(arrayLiteral0);
      astRoot0.addChildToBack(newExpression0);
      ToolErrorReporter toolErrorReporter0 = new ToolErrorReporter(true);
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5_STRICT;
      Config config0 = new Config(linkedHashSet0, linkedHashSet0, true, config_LanguageMode0, true);
      Node node0 = IRFactory.transformTree(astRoot0, (StaticSourceFile) null, "SYNTAX_ERROR_TYPE", config0, toolErrorReporter0);
      assertTrue(node0.isScript());
      assertEquals(1, node0.getLength());
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      LinkedHashSet<String> linkedHashSet0 = new LinkedHashSet<String>();
      Token.CommentType token_CommentType0 = Token.CommentType.LINE;
      Comment comment0 = new Comment(1, 13, token_CommentType0, "280>Aqc");
      SwitchCase switchCase0 = new SwitchCase();
      switchCase0.setJsDocNode(comment0);
      astRoot0.addChildToBack(switchCase0);
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5;
      Config config0 = new Config(linkedHashSet0, linkedHashSet0, false, config_LanguageMode0, false);
      ToolErrorReporter toolErrorReporter0 = new ToolErrorReporter(true);
      Node node0 = IRFactory.transformTree(astRoot0, (StaticSourceFile) null, "language version", config0, toolErrorReporter0);
      assertTrue(node0.isScript());
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      LinkedHashSet<String> linkedHashSet0 = new LinkedHashSet<String>();
      ToolErrorReporter toolErrorReporter0 = new ToolErrorReporter(true);
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5_STRICT;
      Config config0 = new Config(linkedHashSet0, linkedHashSet0, true, config_LanguageMode0, true);
      SimpleSourceFile simpleSourceFile0 = new SimpleSourceFile("error reporter", true);
      Node node0 = IRFactory.transformTree(astRoot0, simpleSourceFile0, "error reporter", config0, toolErrorReporter0);
      assertEquals(1, node0.getLength());
      assertTrue(node0.isFromExterns());
      assertEquals(132, node0.getType());
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      LinkedHashSet<String> linkedHashSet0 = new LinkedHashSet<String>();
      ArrayLiteral arrayLiteral0 = new ArrayLiteral(3, 117);
      astRoot0.addChildToBack(arrayLiteral0);
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5_STRICT;
      Config config0 = new Config(linkedHashSet0, linkedHashSet0, true, config_LanguageMode0, true);
      Context.getCurrentContext();
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      LinkedHashSet<String> linkedHashSet0 = new LinkedHashSet<String>();
      Scope scope0 = new Scope(13, 2);
      astRoot0.addChildToBack(scope0);
      scope0.addChildrenToBack(astRoot0);
      ToolErrorReporter toolErrorReporter0 = new ToolErrorReporter(false);
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5_STRICT;
      Config config0 = new Config(linkedHashSet0, linkedHashSet0, false, config_LanguageMode0, false);
      // Undeclared exception!
      try { 
        IRFactory.transformTree(astRoot0, (StaticSourceFile) null, "Upd+[Pzj%Tw)R6X4*J", config0, toolErrorReporter0);
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
      LinkedHashSet<String> linkedHashSet0 = new LinkedHashSet<String>();
      ArrayLiteral arrayLiteral0 = new ArrayLiteral(2, 22);
      PropertyGet propertyGet0 = new PropertyGet();
      propertyGet0.setTarget(arrayLiteral0);
      astRoot0.addChildToBack(propertyGet0);
      ToolErrorReporter toolErrorReporter0 = new ToolErrorReporter(true);
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5_STRICT;
      Config config0 = new Config(linkedHashSet0, linkedHashSet0, true, config_LanguageMode0, true);
      // Undeclared exception!
      try { 
        IRFactory.transformTree(astRoot0, (StaticSourceFile) null, "error reporter", config0, toolErrorReporter0);
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
      LinkedHashSet<String> linkedHashSet0 = new LinkedHashSet<String>();
      ArrayLiteral arrayLiteral0 = new ArrayLiteral();
      arrayLiteral0.addElement(astRoot0);
      astRoot0.addChildToBack(arrayLiteral0);
      ToolErrorReporter toolErrorReporter0 = new ToolErrorReporter(true);
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT3;
      Config config0 = new Config(linkedHashSet0, linkedHashSet0, true, config_LanguageMode0, true);
      // Undeclared exception!
      try { 
        IRFactory.transformTree(astRoot0, (StaticSourceFile) null, "", config0, toolErrorReporter0);
        fail("Expecting exception: StackOverflowError");
      
      } catch(StackOverflowError e) {
         //
         // no message in exception (getMessage() returned null)
         //
      }
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      LinkedHashSet<String> linkedHashSet0 = new LinkedHashSet<String>();
      BreakStatement breakStatement0 = new BreakStatement(13, 4);
      astRoot0.addChildToBack(breakStatement0);
      ToolErrorReporter toolErrorReporter0 = new ToolErrorReporter(true);
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5;
      Config config0 = new Config(linkedHashSet0, linkedHashSet0, true, config_LanguageMode0, true);
      Node node0 = IRFactory.transformTree(astRoot0, (StaticSourceFile) null, "error reporter", config0, toolErrorReporter0);
      assertEquals(1, node0.getLength());
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      LinkedHashSet<String> linkedHashSet0 = new LinkedHashSet<String>();
      ContinueStatement continueStatement0 = new ContinueStatement(21, 8);
      astRoot0.addChildToBack(continueStatement0);
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT3;
      Config config0 = new Config(linkedHashSet0, linkedHashSet0, false, config_LanguageMode0, false);
      ErrorCollector errorCollector0 = new ErrorCollector();
      Node node0 = IRFactory.transformTree(astRoot0, (StaticSourceFile) null, "language version", config0, errorCollector0);
      assertEquals(0, node0.getLength());
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      LinkedHashSet<String> linkedHashSet0 = new LinkedHashSet<String>();
      Name name0 = new Name();
      ContinueStatement continueStatement0 = new ContinueStatement(name0);
      astRoot0.addChildToBack(continueStatement0);
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT3;
      Config config0 = new Config(linkedHashSet0, linkedHashSet0, false, config_LanguageMode0, true);
      Context.getCurrentContext();
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      LinkedHashSet<String> linkedHashSet0 = new LinkedHashSet<String>();
      GeneratorExpressionLoop generatorExpressionLoop0 = new GeneratorExpressionLoop(8);
      astRoot0.addChildToBack(generatorExpressionLoop0);
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT3;
      Config config0 = new Config(linkedHashSet0, linkedHashSet0, true, config_LanguageMode0, true);
      Context.getCurrentContext();
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      LinkedHashSet<String> linkedHashSet0 = new LinkedHashSet<String>();
      Name name0 = new Name(11, 2, "PiGj983<K+(");
      FunctionNode functionNode0 = new FunctionNode(0, name0);
      astRoot0.addChildToBack(functionNode0);
      ToolErrorReporter toolErrorReporter0 = new ToolErrorReporter(false);
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT3;
      Config config0 = new Config(linkedHashSet0, linkedHashSet0, false, config_LanguageMode0, false);
      // Undeclared exception!
      try { 
        IRFactory.transformTree(astRoot0, (StaticSourceFile) null, "PiGj983<K+(", config0, toolErrorReporter0);
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
      LinkedHashSet<String> linkedHashSet0 = new LinkedHashSet<String>();
      FunctionNode functionNode0 = new FunctionNode(4);
      astRoot0.addChildToBack(functionNode0);
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5_STRICT;
      Config config0 = new Config(linkedHashSet0, linkedHashSet0, false, config_LanguageMode0, false);
      ToolErrorReporter toolErrorReporter0 = new ToolErrorReporter(false);
      Node node0 = IRFactory.transformTree(astRoot0, (StaticSourceFile) null, "language version", config0, toolErrorReporter0);
      assertEquals(0, node0.getLength());
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      LinkedHashSet<String> linkedHashSet0 = new LinkedHashSet<String>(18);
      IfStatement ifStatement0 = new IfStatement();
      Name name0 = new Name(25, "jf(0#");
      PropertyGet propertyGet0 = new PropertyGet(18, 1582, ifStatement0, name0);
      propertyGet0.setTarget(name0);
      astRoot0.addChildToBack(propertyGet0);
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT3;
      Config config0 = new Config(linkedHashSet0, linkedHashSet0, true, config_LanguageMode0, true);
      ErrorCollector errorCollector0 = new ErrorCollector();
      Node node0 = IRFactory.transformTree(astRoot0, (StaticSourceFile) null, "language version", config0, errorCollector0);
      assertEquals(1, node0.getLength());
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      LinkedHashSet<String> linkedHashSet0 = new LinkedHashSet<String>();
      ObjectLiteral objectLiteral0 = new ObjectLiteral();
      astRoot0.addChildToBack(objectLiteral0);
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5_STRICT;
      Config config0 = new Config(linkedHashSet0, linkedHashSet0, true, config_LanguageMode0, true);
      ErrorCollector errorCollector0 = new ErrorCollector();
      Node node0 = IRFactory.transformTree(astRoot0, (StaticSourceFile) null, "error reporter", config0, errorCollector0);
      assertEquals(1, node0.getLength());
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      LinkedHashSet<String> linkedHashSet0 = new LinkedHashSet<String>();
      RegExpLiteral regExpLiteral0 = new RegExpLiteral(6);
      regExpLiteral0.setValue("s5=]gSu=lN\":Rm/')<5");
      astRoot0.addChildToBack(regExpLiteral0);
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT3;
      Config config0 = new Config(linkedHashSet0, linkedHashSet0, false, config_LanguageMode0, false);
      ErrorCollector errorCollector0 = new ErrorCollector();
      Node node0 = IRFactory.transformTree(astRoot0, (StaticSourceFile) null, "s5=]gSu=lN\":Rm/')<5", config0, errorCollector0);
      assertEquals(0, node0.getLength());
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      LinkedHashSet<String> linkedHashSet0 = new LinkedHashSet<String>();
      ReturnStatement returnStatement0 = new ReturnStatement(13);
      astRoot0.addChildToBack(returnStatement0);
      ToolErrorReporter toolErrorReporter0 = new ToolErrorReporter(false);
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5;
      Config config0 = new Config(linkedHashSet0, linkedHashSet0, false, config_LanguageMode0, false);
      Node node0 = IRFactory.transformTree(astRoot0, (StaticSourceFile) null, "language version", config0, toolErrorReporter0);
      assertEquals(0, node0.getLength());
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      LinkedHashSet<String> linkedHashSet0 = new LinkedHashSet<String>();
      ReturnStatement returnStatement0 = new ReturnStatement(16, 8, astRoot0);
      astRoot0.addChildToBack(returnStatement0);
      ToolErrorReporter toolErrorReporter0 = new ToolErrorReporter(true);
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5_STRICT;
      Config config0 = new Config(linkedHashSet0, linkedHashSet0, true, config_LanguageMode0, false);
      // Undeclared exception!
      try { 
        IRFactory.transformTree(astRoot0, (StaticSourceFile) null, "tb;:W}4Pu:Th|nn:", config0, toolErrorReporter0);
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
      LinkedHashSet<String> linkedHashSet0 = new LinkedHashSet<String>(8);
      StringLiteral stringLiteral0 = new StringLiteral();
      stringLiteral0.setValue("getters may not have parameters");
      astRoot0.addChildToBack(stringLiteral0);
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5;
      Config config0 = new Config(linkedHashSet0, linkedHashSet0, false, config_LanguageMode0, false);
      ErrorCollector errorCollector0 = new ErrorCollector();
      Node node0 = IRFactory.transformTree(astRoot0, (StaticSourceFile) null, "getters may not have parameters", config0, errorCollector0);
      assertEquals(0, node0.getLength());
  }

  @Test(timeout = 4000)
  public void test35()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      LinkedHashSet<String> linkedHashSet0 = new LinkedHashSet<String>();
      ArrayLiteral arrayLiteral0 = new ArrayLiteral(1);
      SwitchCase switchCase0 = new SwitchCase(14);
      switchCase0.addStatement(arrayLiteral0);
      astRoot0.addChildToBack(switchCase0);
      ToolErrorReporter toolErrorReporter0 = new ToolErrorReporter(false);
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5;
      Config config0 = new Config(linkedHashSet0, linkedHashSet0, true, config_LanguageMode0, false);
      Node node0 = IRFactory.transformTree(astRoot0, (StaticSourceFile) null, "language version", config0, toolErrorReporter0);
      assertEquals(1, node0.getLength());
  }

  @Test(timeout = 4000)
  public void test36()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      LinkedHashSet<String> linkedHashSet0 = new LinkedHashSet<String>();
      VariableDeclaration variableDeclaration0 = new VariableDeclaration(14, 17);
      astRoot0.addChildToBack(variableDeclaration0);
      ToolErrorReporter toolErrorReporter0 = new ToolErrorReporter(false);
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5_STRICT;
      Config config0 = new Config(linkedHashSet0, linkedHashSet0, true, config_LanguageMode0, false);
      SimpleSourceFile simpleSourceFile0 = new SimpleSourceFile("error reporter", false);
      Node node0 = IRFactory.transformTree(astRoot0, simpleSourceFile0, "error reporter", config0, toolErrorReporter0);
      assertEquals(1, node0.getLength());
  }

  @Test(timeout = 4000)
  public void test37()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      LinkedHashSet<String> linkedHashSet0 = new LinkedHashSet<String>();
      FunctionCall functionCall0 = new FunctionCall(9, 19);
      astRoot0.addChildToBack(functionCall0);
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT3;
      Config config0 = new Config(linkedHashSet0, linkedHashSet0, true, config_LanguageMode0, true);
      ToolErrorReporter toolErrorReporter0 = new ToolErrorReporter(true);
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
  public void test38()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      LinkedHashSet<String> linkedHashSet0 = new LinkedHashSet<String>();
      ExpressionStatement expressionStatement0 = new ExpressionStatement();
      astRoot0.addChildToBack(expressionStatement0);
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5_STRICT;
      Config config0 = new Config(linkedHashSet0, linkedHashSet0, true, config_LanguageMode0, true);
      ErrorCollector errorCollector0 = new ErrorCollector();
      // Undeclared exception!
      try { 
        IRFactory.transformTree(astRoot0, (StaticSourceFile) null, "error reporter", config0, errorCollector0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
      }
  }
}