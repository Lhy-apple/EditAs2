/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 02:10:44 GMT 2023
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
import com.google.javascript.rhino.head.ast.ForInLoop;
import com.google.javascript.rhino.head.ast.ForLoop;
import com.google.javascript.rhino.head.ast.FunctionCall;
import com.google.javascript.rhino.head.ast.FunctionNode;
import com.google.javascript.rhino.head.ast.InfixExpression;
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
import com.google.javascript.rhino.head.ast.XmlPropRef;
import com.google.javascript.rhino.head.tools.ToolErrorReporter;
import com.google.javascript.rhino.jstype.SimpleSourceFile;
import com.google.javascript.rhino.jstype.StaticSourceFile;
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
      SimpleSourceFile simpleSourceFile0 = new SimpleSourceFile("5ZoSj{ra,+_`Qc", true);
      LinkedHashSet<String> linkedHashSet0 = new LinkedHashSet<String>();
      ConditionalExpression conditionalExpression0 = new ConditionalExpression();
      astRoot0.addChild(conditionalExpression0);
      Context context0 = Context.getCurrentContext();
      ErrorReporter errorReporter0 = context0.getErrorReporter();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT3;
      Config config0 = new Config(linkedHashSet0, linkedHashSet0, false, config_LanguageMode0, true);
      // Undeclared exception!
      try { 
        IRFactory.transformTree(astRoot0, simpleSourceFile0, "language version", config0, errorReporter0);
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
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT3;
      Config config0 = new Config(linkedHashSet0, linkedHashSet0, true, config_LanguageMode0, true);
      NumberLiteral numberLiteral0 = new NumberLiteral(5, "6gUoWbds", 13);
      ParenthesizedExpression parenthesizedExpression0 = new ParenthesizedExpression(numberLiteral0);
      astRoot0.addChild(parenthesizedExpression0);
      ToolErrorReporter toolErrorReporter0 = new ToolErrorReporter(true);
      SimpleSourceFile simpleSourceFile0 = new SimpleSourceFile("wIzw", true);
      Node node0 = IRFactory.transformTree(astRoot0, simpleSourceFile0, "'(`f4$jX'wSf(YyNo", config0, toolErrorReporter0);
      assertTrue(node0.isScript());
      assertEquals(14, node0.getLength());
      assertTrue(node0.isFromExterns());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      LinkedHashSet<String> linkedHashSet0 = new LinkedHashSet<String>();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT3;
      Config config0 = new Config(linkedHashSet0, linkedHashSet0, true, config_LanguageMode0, true);
      DoLoop doLoop0 = new DoLoop(1);
      astRoot0.addChild(doLoop0);
      SimpleSourceFile simpleSourceFile0 = new SimpleSourceFile("Bad initialBufferCapacity: ", false);
      // Undeclared exception!
      try { 
        IRFactory.transformTree(astRoot0, simpleSourceFile0, "Bad initialBufferCapacity: ", config0, (ErrorReporter) null);
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
      SimpleSourceFile simpleSourceFile0 = new SimpleSourceFile("implements", true);
      LinkedHashSet<String> linkedHashSet0 = new LinkedHashSet<String>();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5_STRICT;
      Config config0 = new Config(linkedHashSet0, linkedHashSet0, true, config_LanguageMode0, false);
      ToolErrorReporter toolErrorReporter0 = new ToolErrorReporter(false);
      Name name0 = new Name(0, "implements");
      PropertyGet propertyGet0 = new PropertyGet(name0, name0, (-306));
      astRoot0.addChild(propertyGet0);
      Node node0 = IRFactory.transformTree(astRoot0, simpleSourceFile0, "8mCHEx@u", config0, toolErrorReporter0);
      assertEquals(11, node0.getLength());
      assertTrue(node0.isFromExterns());
      assertTrue(node0.isScript());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      SimpleSourceFile simpleSourceFile0 = new SimpleSourceFile("7TuD~^0=", false);
      LinkedHashSet<String> linkedHashSet0 = new LinkedHashSet<String>();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5;
      Config config0 = new Config(linkedHashSet0, linkedHashSet0, false, config_LanguageMode0, false);
      EmptyExpression emptyExpression0 = new EmptyExpression(1338, 142);
      astRoot0.addChild(emptyExpression0);
      ToolErrorReporter toolErrorReporter0 = new ToolErrorReporter(false);
      Node node0 = IRFactory.transformTree(astRoot0, simpleSourceFile0, "i{'\"~XQB0\"NB0Tq", config0, toolErrorReporter0);
      assertEquals(0, node0.getLength());
      assertTrue(node0.isScript());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      SimpleSourceFile simpleSourceFile0 = new SimpleSourceFile("(jsConstructor", true);
      LinkedHashSet<String> linkedHashSet0 = new LinkedHashSet<String>();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5;
      Config config0 = new Config(linkedHashSet0, linkedHashSet0, false, config_LanguageMode0, true);
      ToolErrorReporter toolErrorReporter0 = new ToolErrorReporter(true);
      ForInLoop forInLoop0 = new ForInLoop(16);
      astRoot0.addChild(forInLoop0);
      // Undeclared exception!
      try { 
        IRFactory.transformTree(astRoot0, simpleSourceFile0, "(jsConstructor", config0, toolErrorReporter0);
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
      SimpleSourceFile simpleSourceFile0 = new SimpleSourceFile("Bad initialBufferCapacity: ", false);
      LinkedHashSet<String> linkedHashSet0 = new LinkedHashSet<String>();
      ElementGet elementGet0 = new ElementGet(0, 2);
      astRoot0.addChild(elementGet0);
      ErrorCollector errorCollector0 = new ErrorCollector();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5_STRICT;
      Config config0 = new Config(linkedHashSet0, linkedHashSet0, true, config_LanguageMode0, false);
      // Undeclared exception!
      try { 
        IRFactory.transformTree(astRoot0, simpleSourceFile0, "error reporter", config0, errorCollector0);
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
      SimpleSourceFile simpleSourceFile0 = new SimpleSourceFile("7TuD~^0=", true);
      LinkedHashSet<String> linkedHashSet0 = new LinkedHashSet<String>();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT3;
      Config config0 = new Config(linkedHashSet0, linkedHashSet0, true, config_LanguageMode0, true);
      ObjectProperty objectProperty0 = new ObjectProperty();
      astRoot0.addChild(objectProperty0);
      ToolErrorReporter toolErrorReporter0 = new ToolErrorReporter(true);
      // Undeclared exception!
      try { 
        IRFactory.transformTree(astRoot0, simpleSourceFile0, "7TuD~^0=", config0, toolErrorReporter0);
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
      SimpleSourceFile simpleSourceFile0 = new SimpleSourceFile("jsHrConsthctr", false);
      LinkedHashSet<String> linkedHashSet0 = new LinkedHashSet<String>();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5_STRICT;
      Config config0 = new Config(linkedHashSet0, linkedHashSet0, false, config_LanguageMode0, true);
      WithStatement withStatement0 = new WithStatement();
      astRoot0.addChild(withStatement0);
      ToolErrorReporter toolErrorReporter0 = new ToolErrorReporter(false);
      // Undeclared exception!
      try { 
        IRFactory.transformTree(astRoot0, simpleSourceFile0, "wIzw", config0, toolErrorReporter0);
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
      SimpleSourceFile simpleSourceFile0 = new SimpleSourceFile("jsConstructovr", false);
      LinkedHashSet<String> linkedHashSet0 = new LinkedHashSet<String>();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5_STRICT;
      Config config0 = new Config(linkedHashSet0, linkedHashSet0, false, config_LanguageMode0, false);
      ForLoop forLoop0 = new ForLoop();
      astRoot0.addChild(forLoop0);
      ToolErrorReporter toolErrorReporter0 = new ToolErrorReporter(false);
      // Undeclared exception!
      try { 
        IRFactory.transformTree(astRoot0, simpleSourceFile0, " DH?yw@", config0, toolErrorReporter0);
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
      SimpleSourceFile simpleSourceFile0 = new SimpleSourceFile("Bad initialBufferCapacity: ", true);
      LinkedHashSet<String> linkedHashSet0 = new LinkedHashSet<String>();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT3;
      Config config0 = new Config(linkedHashSet0, linkedHashSet0, true, config_LanguageMode0, true);
      Label label0 = new Label(2, 1);
      astRoot0.addChild(label0);
      ToolErrorReporter toolErrorReporter0 = new ToolErrorReporter(true);
      // Undeclared exception!
      try { 
        IRFactory.transformTree(astRoot0, simpleSourceFile0, "A\"+b#Jo", config0, toolErrorReporter0);
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
      SimpleSourceFile simpleSourceFile0 = new SimpleSourceFile("jsConstructor", true);
      LinkedHashSet<String> linkedHashSet0 = new LinkedHashSet<String>();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT3;
      Config config0 = new Config(linkedHashSet0, linkedHashSet0, true, config_LanguageMode0, true);
      AstRoot astRoot0 = new AstRoot();
      NewExpression newExpression0 = new NewExpression(0, 1);
      astRoot0.addChild(newExpression0);
      Context context0 = Context.enter();
      ErrorReporter errorReporter0 = context0.getErrorReporter();
      // Undeclared exception!
      try { 
        IRFactory.transformTree(astRoot0, simpleSourceFile0, "g\"iw2)mPH9<", config0, errorReporter0);
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
      SimpleSourceFile simpleSourceFile0 = new SimpleSourceFile("O}{NV5jkRR;mV{", false);
      LinkedHashSet<String> linkedHashSet0 = new LinkedHashSet<String>();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5;
      Config config0 = new Config(linkedHashSet0, linkedHashSet0, false, config_LanguageMode0, false);
      ExpressionStatement expressionStatement0 = new ExpressionStatement(astRoot0, true);
      astRoot0.addChild(expressionStatement0);
      ToolErrorReporter toolErrorReporter0 = new ToolErrorReporter(true);
      // Undeclared exception!
      try { 
        IRFactory.transformTree(astRoot0, simpleSourceFile0, "error reporter", config0, toolErrorReporter0);
        fail("Expecting exception: StackOverflowError");
      
      } catch(StackOverflowError e) {
         //
         // no message in exception (getMessage() returned null)
         //
      }
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      SimpleSourceFile simpleSourceFile0 = new SimpleSourceFile("jsConstructor", true);
      LinkedHashSet<String> linkedHashSet0 = new LinkedHashSet<String>();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT3;
      Config config0 = new Config(linkedHashSet0, linkedHashSet0, true, config_LanguageMode0, true);
      ToolErrorReporter toolErrorReporter0 = new ToolErrorReporter(true);
      AstRoot astRoot0 = new AstRoot();
      Scope scope0 = new Scope();
      astRoot0.addChild(scope0);
      Node node0 = IRFactory.transformTree(astRoot0, simpleSourceFile0, "dUcA~GW{", config0, toolErrorReporter0);
      assertTrue(node0.isScript());
      assertTrue(node0.isFromExterns());
      assertEquals(1, node0.getLength());
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      SimpleSourceFile simpleSourceFile0 = new SimpleSourceFile("jsConstructor", true);
      LinkedHashSet<String> linkedHashSet0 = new LinkedHashSet<String>();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5;
      Config config0 = new Config(linkedHashSet0, linkedHashSet0, true, config_LanguageMode0, true);
      ToolErrorReporter toolErrorReporter0 = new ToolErrorReporter(true);
      AstRoot astRoot0 = new AstRoot();
      WhileLoop whileLoop0 = new WhileLoop();
      astRoot0.addChild(whileLoop0);
      // Undeclared exception!
      try { 
        IRFactory.transformTree(astRoot0, simpleSourceFile0, "jsConstructor", config0, toolErrorReporter0);
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
      SimpleSourceFile simpleSourceFile0 = new SimpleSourceFile("Bad initialBufferCapacity: ", true);
      LinkedHashSet<String> linkedHashSet0 = new LinkedHashSet<String>();
      Block block0 = new Block();
      astRoot0.addChild(block0);
      Context context0 = Context.getCurrentContext();
      ErrorReporter errorReporter0 = context0.getErrorReporter();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5_STRICT;
      Config config0 = new Config(linkedHashSet0, linkedHashSet0, false, config_LanguageMode0, true);
      Node node0 = IRFactory.transformTree(astRoot0, simpleSourceFile0, "Bad initialBufferCapacity: ", config0, errorReporter0);
      assertEquals(0, node0.getLength());
      assertTrue(node0.isFromExterns());
      assertEquals(132, node0.getType());
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      SimpleSourceFile simpleSourceFile0 = new SimpleSourceFile("Bad initiclBufferCapacity: ", true);
      LinkedHashSet<String> linkedHashSet0 = new LinkedHashSet<String>();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5;
      Config config0 = new Config(linkedHashSet0, linkedHashSet0, true, config_LanguageMode0, true);
      ToolErrorReporter toolErrorReporter0 = new ToolErrorReporter(true);
      XmlPropRef xmlPropRef0 = new XmlPropRef(1, 0);
      astRoot0.addChild(xmlPropRef0);
      Node node0 = IRFactory.transformTree(astRoot0, simpleSourceFile0, "u8!up-pG>6 Ev", config0, toolErrorReporter0);
      assertTrue(node0.isFromExterns());
      assertTrue(node0.isScript());
      assertEquals(2, node0.getLength());
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      SimpleSourceFile simpleSourceFile0 = new SimpleSourceFile("7TuD~^0=", false);
      LinkedHashSet<String> linkedHashSet0 = new LinkedHashSet<String>();
      ThrowStatement throwStatement0 = new ThrowStatement();
      astRoot0.addChild(throwStatement0);
      TreeSet<String> treeSet0 = new TreeSet<String>();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT3;
      Config config0 = new Config(linkedHashSet0, treeSet0, false, config_LanguageMode0, false);
      Context context0 = Context.enter();
      ErrorReporter errorReporter0 = context0.getErrorReporter();
      // Undeclared exception!
      try { 
        IRFactory.transformTree(astRoot0, simpleSourceFile0, "7TuD~^0=", config0, errorReporter0);
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
      SimpleSourceFile simpleSourceFile0 = new SimpleSourceFile("jsConstructovr", false);
      LinkedHashSet<String> linkedHashSet0 = new LinkedHashSet<String>();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT3;
      Config config0 = new Config(linkedHashSet0, linkedHashSet0, false, config_LanguageMode0, false);
      ToolErrorReporter toolErrorReporter0 = new ToolErrorReporter(false);
      RegExpLiteral regExpLiteral0 = new RegExpLiteral();
      astRoot0.addChild(regExpLiteral0);
      regExpLiteral0.setValue("jsConstructovr");
      Node node0 = IRFactory.transformTree(astRoot0, simpleSourceFile0, "jsConstructovr", config0, toolErrorReporter0);
      assertEquals(0, node0.getLength());
      assertEquals(132, node0.getType());
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      LinkedHashSet<String> linkedHashSet0 = new LinkedHashSet<String>();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5_STRICT;
      Config config0 = new Config(linkedHashSet0, linkedHashSet0, true, config_LanguageMode0, true);
      BreakStatement breakStatement0 = new BreakStatement(4);
      Token.CommentType token_CommentType0 = Token.CommentType.LINE;
      Comment comment0 = new Comment(9, 9, token_CommentType0, "language version");
      breakStatement0.setJsDocNode(comment0);
      astRoot0.addChild(breakStatement0);
      ToolErrorReporter toolErrorReporter0 = new ToolErrorReporter(true);
      SimpleSourceFile simpleSourceFile0 = new SimpleSourceFile("language version", true);
      Node node0 = IRFactory.transformTree(astRoot0, simpleSourceFile0, "language version", config0, toolErrorReporter0);
      assertTrue(node0.isFromExterns());
      assertEquals(6, node0.getLength());
      assertEquals(132, node0.getType());
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      Context context0 = Context.enter();
      ErrorReporter errorReporter0 = context0.getErrorReporter();
      // Undeclared exception!
      try { 
        IRFactory.transformTree(astRoot0, (StaticSourceFile) null, "", (Config) null, errorReporter0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.parsing.IRFactory", e);
      }
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      SimpleSourceFile simpleSourceFile0 = new SimpleSourceFile("u", false);
      LinkedHashSet<String> linkedHashSet0 = new LinkedHashSet<String>();
      SwitchCase switchCase0 = new SwitchCase(11, 24);
      astRoot0.addChild(switchCase0);
      ErrorCollector errorCollector0 = new ErrorCollector();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5_STRICT;
      Config config0 = new Config(linkedHashSet0, linkedHashSet0, false, config_LanguageMode0, true);
      Node node0 = IRFactory.transformTree(astRoot0, simpleSourceFile0, "\n * @", config0, errorCollector0);
      assertTrue(node0.isScript());
      assertEquals(0, node0.getLength());
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      SimpleSourceFile simpleSourceFile0 = new SimpleSourceFile("5ZoSj{ra,+_`Qc", true);
      LinkedHashSet<String> linkedHashSet0 = new LinkedHashSet<String>();
      ObjectLiteral objectLiteral0 = new ObjectLiteral();
      ObjectProperty objectProperty0 = new ObjectProperty((-2703));
      objectLiteral0.addElement(objectProperty0);
      astRoot0.addChild(objectLiteral0);
      ErrorCollector errorCollector0 = new ErrorCollector();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5_STRICT;
      Config config0 = new Config(linkedHashSet0, linkedHashSet0, true, config_LanguageMode0, true);
      // Undeclared exception!
      try { 
        IRFactory.transformTree(astRoot0, simpleSourceFile0, "language version", config0, errorCollector0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
      }
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      LinkedHashSet<String> linkedHashSet0 = new LinkedHashSet<String>();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT3;
      Config config0 = new Config(linkedHashSet0, linkedHashSet0, true, config_LanguageMode0, true);
      ArrayLiteral arrayLiteral0 = new ArrayLiteral(14, 153);
      astRoot0.addChild(arrayLiteral0);
      ToolErrorReporter toolErrorReporter0 = new ToolErrorReporter(true);
      SimpleSourceFile simpleSourceFile0 = new SimpleSourceFile("msg.jsdoc.preservertry", true);
      Node node0 = IRFactory.transformTree(astRoot0, simpleSourceFile0, "msg.jsdoc.preservertry", config0, toolErrorReporter0);
      assertEquals(168, node0.getLength());
      assertEquals(132, node0.getType());
      assertTrue(node0.isFromExterns());
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      LinkedHashSet<String> linkedHashSet0 = new LinkedHashSet<String>();
      ErrorCollector errorCollector0 = new ErrorCollector();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5;
      Config config0 = new Config(linkedHashSet0, linkedHashSet0, true, config_LanguageMode0, true);
      SimpleSourceFile simpleSourceFile0 = new SimpleSourceFile("language version", true);
      Node node0 = IRFactory.transformTree(astRoot0, simpleSourceFile0, "error reporter", config0, errorCollector0);
      assertTrue(node0.isScript());
      assertTrue(node0.isFromExterns());
      assertEquals(1, node0.getLength());
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      SimpleSourceFile simpleSourceFile0 = new SimpleSourceFile("jsConstructor", true);
      LinkedHashSet<String> linkedHashSet0 = new LinkedHashSet<String>();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5;
      Config config0 = new Config(linkedHashSet0, linkedHashSet0, true, config_LanguageMode0, true);
      ToolErrorReporter toolErrorReporter0 = new ToolErrorReporter(true);
      AstRoot astRoot0 = new AstRoot();
      FunctionNode functionNode0 = new FunctionNode(2);
      astRoot0.addChild(functionNode0);
      Node node0 = IRFactory.transformTree(astRoot0, simpleSourceFile0, "jsConstructor", config0, toolErrorReporter0);
      assertEquals(4, node0.getLength());
      assertEquals(132, node0.getType());
      assertTrue(node0.isFromExterns());
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      SimpleSourceFile simpleSourceFile0 = new SimpleSourceFile("Bad initialBufferCapacity: ", true);
      LinkedHashSet<String> linkedHashSet0 = new LinkedHashSet<String>();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT3;
      Config config0 = new Config(linkedHashSet0, linkedHashSet0, true, config_LanguageMode0, true);
      ContinueStatement continueStatement0 = new ContinueStatement(8);
      astRoot0.addChild(continueStatement0);
      Node node0 = IRFactory.transformTree(astRoot0, simpleSourceFile0, "Bad initialBufferCapacity: ", config0, (ErrorReporter) null);
      assertEquals(8, node0.getLength());
      assertTrue(node0.isScript());
      assertTrue(node0.isFromExterns());
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      SimpleSourceFile simpleSourceFile0 = new SimpleSourceFile("", false);
      LinkedHashSet<String> linkedHashSet0 = new LinkedHashSet<String>();
      Name name0 = new Name(2, 25);
      ContinueStatement continueStatement0 = new ContinueStatement(139, 4, name0);
      ExpressionStatement expressionStatement0 = new ExpressionStatement(continueStatement0);
      astRoot0.addChild(expressionStatement0);
      Context context0 = Context.getCurrentContext();
      ErrorReporter errorReporter0 = context0.getErrorReporter();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT3;
      Config config0 = new Config(linkedHashSet0, linkedHashSet0, false, config_LanguageMode0, true);
      // Undeclared exception!
      try { 
        IRFactory.transformTree(astRoot0, simpleSourceFile0, "", config0, errorReporter0);
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
      SimpleSourceFile simpleSourceFile0 = new SimpleSourceFile("jsConstructor", true);
      LinkedHashSet<String> linkedHashSet0 = new LinkedHashSet<String>();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT3;
      Config config0 = new Config(linkedHashSet0, linkedHashSet0, true, config_LanguageMode0, true);
      ToolErrorReporter toolErrorReporter0 = new ToolErrorReporter(true);
      AstRoot astRoot0 = new AstRoot();
      Name name0 = new Name(2, "jsConstructor");
      FunctionNode functionNode0 = new FunctionNode(16, name0);
      astRoot0.addChild(functionNode0);
      // Undeclared exception!
      try { 
        IRFactory.transformTree(astRoot0, simpleSourceFile0, "jsConstructor", config0, toolErrorReporter0);
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
      SimpleSourceFile simpleSourceFile0 = new SimpleSourceFile("Bad initialBufferCapacity: ", false);
      LinkedHashSet<String> linkedHashSet0 = new LinkedHashSet<String>();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5;
      Config config0 = new Config(linkedHashSet0, linkedHashSet0, false, config_LanguageMode0, false);
      LabeledStatement labeledStatement0 = new LabeledStatement(6, 20);
      astRoot0.addChild(labeledStatement0);
      ToolErrorReporter toolErrorReporter0 = new ToolErrorReporter(false);
      // Undeclared exception!
      try { 
        IRFactory.transformTree(astRoot0, simpleSourceFile0, "Bad initialBufferCapacity: ", config0, toolErrorReporter0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
      }
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      SimpleSourceFile simpleSourceFile0 = new SimpleSourceFile("(jsConstructor", true);
      LinkedHashSet<String> linkedHashSet0 = new LinkedHashSet<String>();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5;
      Config config0 = new Config(linkedHashSet0, linkedHashSet0, false, config_LanguageMode0, true);
      ToolErrorReporter toolErrorReporter0 = new ToolErrorReporter(true);
      Name name0 = new Name(15, "(jsConstructor");
      PropertyGet propertyGet0 = new PropertyGet(name0, name0, 19);
      astRoot0.addChild(propertyGet0);
      Node node0 = IRFactory.transformTree(astRoot0, simpleSourceFile0, "(jsConstructor", config0, toolErrorReporter0);
      assertEquals(0, node0.getLength());
      assertTrue(node0.isFromExterns());
      assertEquals(132, node0.getType());
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      SimpleSourceFile simpleSourceFile0 = new SimpleSourceFile("5ZoSj{ra,+_`Qc", true);
      LinkedHashSet<String> linkedHashSet0 = new LinkedHashSet<String>();
      ObjectLiteral objectLiteral0 = new ObjectLiteral();
      astRoot0.addChild(objectLiteral0);
      ErrorCollector errorCollector0 = new ErrorCollector();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5_STRICT;
      Config config0 = new Config(linkedHashSet0, linkedHashSet0, true, config_LanguageMode0, true);
      Node node0 = IRFactory.transformTree(astRoot0, simpleSourceFile0, "language version", config0, errorCollector0);
      assertEquals(1, node0.getLength());
      assertTrue(node0.isFromExterns());
      assertTrue(node0.isScript());
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      SimpleSourceFile simpleSourceFile0 = new SimpleSourceFile("jsConstrctor", false);
      LinkedHashSet<String> linkedHashSet0 = new LinkedHashSet<String>();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5;
      Config config0 = new Config(linkedHashSet0, linkedHashSet0, false, config_LanguageMode0, false);
      ReturnStatement returnStatement0 = new ReturnStatement(4);
      astRoot0.addChild(returnStatement0);
      ToolErrorReporter toolErrorReporter0 = new ToolErrorReporter(false);
      Node node0 = IRFactory.transformTree(astRoot0, simpleSourceFile0, "jsConstrctor", config0, toolErrorReporter0);
      assertTrue(node0.isScript());
      assertEquals(0, node0.getLength());
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      SimpleSourceFile simpleSourceFile0 = new SimpleSourceFile("jsConstrctor", false);
      LinkedHashSet<String> linkedHashSet0 = new LinkedHashSet<String>();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5;
      Config config0 = new Config(linkedHashSet0, linkedHashSet0, false, config_LanguageMode0, false);
      ReturnStatement returnStatement0 = new ReturnStatement(4);
      returnStatement0.setReturnValue(astRoot0);
      astRoot0.addChild(returnStatement0);
      ToolErrorReporter toolErrorReporter0 = new ToolErrorReporter(false);
      // Undeclared exception!
      try { 
        IRFactory.transformTree(astRoot0, simpleSourceFile0, "jsConstrctor", config0, toolErrorReporter0);
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
      VariableDeclaration variableDeclaration0 = new VariableDeclaration();
      astRoot0.addChild(variableDeclaration0);
      SimpleSourceFile simpleSourceFile0 = new SimpleSourceFile("@[F5aSbpk;U2\"G[~D>", false);
      LinkedHashSet<String> linkedHashSet0 = new LinkedHashSet<String>();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5;
      Config config0 = new Config(linkedHashSet0, linkedHashSet0, true, config_LanguageMode0, true);
      ToolErrorReporter toolErrorReporter0 = new ToolErrorReporter(false);
      Node node0 = IRFactory.transformTree(astRoot0, simpleSourceFile0, "@[F5aSbpk;U2\"G[~D>", config0, toolErrorReporter0);
      assertEquals(1, node0.getLength());
      assertEquals(132, node0.getType());
  }

  @Test(timeout = 4000)
  public void test35()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      VariableDeclaration variableDeclaration0 = new VariableDeclaration();
      astRoot0.addChild(variableDeclaration0);
      SimpleSourceFile simpleSourceFile0 = new SimpleSourceFile("jsHrConsthctr", false);
      LinkedHashSet<String> linkedHashSet0 = new LinkedHashSet<String>();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT3;
      Config config0 = new Config(linkedHashSet0, linkedHashSet0, false, config_LanguageMode0, false);
      ToolErrorReporter toolErrorReporter0 = new ToolErrorReporter(false);
      Node node0 = IRFactory.transformTree(astRoot0, simpleSourceFile0, "jsHrConsthctr", config0, toolErrorReporter0);
      assertEquals(0, node0.getLength());
      assertTrue(node0.isScript());
  }

  @Test(timeout = 4000)
  public void test36()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      SimpleSourceFile simpleSourceFile0 = new SimpleSourceFile("jsConstructor", true);
      LinkedHashSet<String> linkedHashSet0 = new LinkedHashSet<String>();
      Context context0 = Context.getCurrentContext();
      InfixExpression infixExpression0 = new InfixExpression(22, astRoot0, astRoot0, 0);
      astRoot0.addChild(infixExpression0);
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5;
      Config config0 = new Config(linkedHashSet0, linkedHashSet0, true, config_LanguageMode0, false);
      ErrorReporter errorReporter0 = context0.getErrorReporter();
      // Undeclared exception!
      try { 
        IRFactory.transformTree(astRoot0, simpleSourceFile0, "language version", config0, errorReporter0);
        fail("Expecting exception: StackOverflowError");
      
      } catch(StackOverflowError e) {
         //
         // no message in exception (getMessage() returned null)
         //
      }
  }

  @Test(timeout = 4000)
  public void test37()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      LinkedHashSet<String> linkedHashSet0 = new LinkedHashSet<String>();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5;
      Config config0 = new Config(linkedHashSet0, linkedHashSet0, true, config_LanguageMode0, false);
      FunctionCall functionCall0 = new FunctionCall();
      astRoot0.addChild(functionCall0);
      SimpleSourceFile simpleSourceFile0 = new SimpleSourceFile("Bad initialBufferCapacity: ", false);
      // Undeclared exception!
      try { 
        IRFactory.transformTree(astRoot0, simpleSourceFile0, "Bad initialBufferCapacity: ", config0, (ErrorReporter) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
      }
  }
}