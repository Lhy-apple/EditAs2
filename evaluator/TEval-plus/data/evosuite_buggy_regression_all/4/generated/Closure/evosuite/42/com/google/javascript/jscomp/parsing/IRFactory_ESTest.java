/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 21:12:31 GMT 2023
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
import com.google.javascript.rhino.head.ast.ArrayLiteral;
import com.google.javascript.rhino.head.ast.AstRoot;
import com.google.javascript.rhino.head.ast.Block;
import com.google.javascript.rhino.head.ast.Comment;
import com.google.javascript.rhino.head.ast.DoLoop;
import com.google.javascript.rhino.head.ast.ElementGet;
import com.google.javascript.rhino.head.ast.EmptyExpression;
import com.google.javascript.rhino.head.ast.ErrorCollector;
import com.google.javascript.rhino.head.ast.ExpressionStatement;
import com.google.javascript.rhino.head.ast.ForInLoop;
import com.google.javascript.rhino.head.ast.ForLoop;
import com.google.javascript.rhino.head.ast.FunctionCall;
import com.google.javascript.rhino.head.ast.FunctionNode;
import com.google.javascript.rhino.head.ast.KeywordLiteral;
import com.google.javascript.rhino.head.ast.LabeledStatement;
import com.google.javascript.rhino.head.ast.Name;
import com.google.javascript.rhino.head.ast.NewExpression;
import com.google.javascript.rhino.head.ast.NumberLiteral;
import com.google.javascript.rhino.head.ast.ObjectLiteral;
import com.google.javascript.rhino.head.ast.PropertyGet;
import com.google.javascript.rhino.head.ast.ReturnStatement;
import com.google.javascript.rhino.head.ast.Scope;
import com.google.javascript.rhino.head.ast.StringLiteral;
import com.google.javascript.rhino.head.ast.SwitchCase;
import com.google.javascript.rhino.head.ast.ThrowStatement;
import com.google.javascript.rhino.head.ast.VariableDeclaration;
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
      ErrorCollector errorCollector0 = new ErrorCollector();
      DoLoop doLoop0 = new DoLoop(140);
      astRoot0.addChildToFront(doLoop0);
      TreeSet<String> treeSet0 = new TreeSet<String>();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT3;
      Config config0 = new Config(treeSet0, treeSet0, true, config_LanguageMode0, true);
      SimpleSourceFile simpleSourceFile0 = new SimpleSourceFile("error reporter", false);
      // Undeclared exception!
      try { 
        IRFactory.transformTree(astRoot0, simpleSourceFile0, "c`|CYlj", config0, errorCollector0);
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
      ErrorCollector errorCollector0 = new ErrorCollector();
      TreeSet<String> treeSet0 = new TreeSet<String>();
      PropertyGet propertyGet0 = new PropertyGet(1);
      astRoot0.addChildToFront(propertyGet0);
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5_STRICT;
      Config config0 = new Config(treeSet0, treeSet0, true, config_LanguageMode0, true);
      SimpleSourceFile simpleSourceFile0 = new SimpleSourceFile("Sc=+[UOz", true);
      // Undeclared exception!
      try { 
        IRFactory.transformTree(astRoot0, simpleSourceFile0, "_YbGI(Y3O0Zg=nH/0", config0, errorCollector0);
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
      ErrorCollector errorCollector0 = new ErrorCollector();
      EmptyExpression emptyExpression0 = new EmptyExpression(3, 18);
      astRoot0.addChildToFront(emptyExpression0);
      Locale locale0 = Locale.UK;
      Set<String> set0 = locale0.getUnicodeLocaleAttributes();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5_STRICT;
      Config config0 = new Config(set0, set0, false, config_LanguageMode0, true);
      Node node0 = IRFactory.transformTree(astRoot0, (StaticSourceFile) null, "Unexpected opcode for 1 operand", config0, errorCollector0);
      assertEquals(0, node0.getLength());
      assertTrue(node0.isScript());
      assertTrue(node0.hasChildren());
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      ErrorCollector errorCollector0 = new ErrorCollector();
      ForInLoop forInLoop0 = new ForInLoop(18);
      astRoot0.addChildToFront(forInLoop0);
      TreeSet<String> treeSet0 = new TreeSet<String>();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT3;
      Config config0 = new Config(treeSet0, treeSet0, false, config_LanguageMode0, false);
      SimpleSourceFile simpleSourceFile0 = new SimpleSourceFile("\"hh[", true);
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
  public void test04()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      ErrorCollector errorCollector0 = new ErrorCollector();
      TreeSet<String> treeSet0 = new TreeSet<String>();
      ObjectLiteral objectLiteral0 = new ObjectLiteral();
      ElementGet elementGet0 = new ElementGet(astRoot0, objectLiteral0);
      astRoot0.addChildToFront(elementGet0);
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5_STRICT;
      Config config0 = new Config(treeSet0, treeSet0, true, config_LanguageMode0, true);
      SimpleSourceFile simpleSourceFile0 = new SimpleSourceFile("21.hmZ'5w0U`Z`;dw#", true);
      // Undeclared exception!
      try { 
        IRFactory.transformTree(astRoot0, simpleSourceFile0, "language version", config0, errorCollector0);
        fail("Expecting exception: StackOverflowError");
      
      } catch(StackOverflowError e) {
         //
         // no message in exception (getMessage() returned null)
         //
      }
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      ErrorCollector errorCollector0 = new ErrorCollector();
      WithStatement withStatement0 = new WithStatement(7, 7);
      astRoot0.addChildToFront(withStatement0);
      TreeSet<String> treeSet0 = new TreeSet<String>();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5_STRICT;
      Config config0 = new Config(treeSet0, treeSet0, true, config_LanguageMode0, false);
      SimpleSourceFile simpleSourceFile0 = new SimpleSourceFile("language version", false);
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
  public void test06()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      ErrorCollector errorCollector0 = new ErrorCollector();
      TreeSet<String> treeSet0 = new TreeSet<String>();
      ForLoop forLoop0 = new ForLoop();
      astRoot0.addChildToFront(forLoop0);
      LinkedHashSet<String> linkedHashSet0 = new LinkedHashSet<String>();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5;
      Config config0 = new Config(treeSet0, linkedHashSet0, false, config_LanguageMode0, false);
      // Undeclared exception!
      try { 
        IRFactory.transformTree(astRoot0, (StaticSourceFile) null, "msg.bad.switch", config0, errorCollector0);
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
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5;
      NewExpression newExpression0 = new NewExpression(5);
      astRoot0.addChildToFront(newExpression0);
      Config config0 = new Config(treeSet0, treeSet0, true, config_LanguageMode0, true);
      // Undeclared exception!
      try { 
        IRFactory.transformTree(astRoot0, (StaticSourceFile) null, "error reporter", config0, (ErrorReporter) null);
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
      NumberLiteral numberLiteral0 = new NumberLiteral();
      astRoot0.addChildrenToBack(numberLiteral0);
      ErrorCollector errorCollector0 = new ErrorCollector();
      Locale locale0 = Locale.KOREA;
      Set<String> set0 = locale0.getUnicodeLocaleAttributes();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT3;
      Config config0 = new Config(set0, set0, true, config_LanguageMode0, true);
      SimpleSourceFile simpleSourceFile0 = new SimpleSourceFile("error reporter", true);
      Node node0 = IRFactory.transformTree(astRoot0, simpleSourceFile0, "error reporter", config0, errorCollector0);
      assertEquals(1, node0.getLength());
      assertEquals(132, node0.getType());
      assertTrue(node0.isFromExterns());
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      ErrorCollector errorCollector0 = new ErrorCollector();
      ExpressionStatement expressionStatement0 = new ExpressionStatement(astRoot0);
      astRoot0.addChildToFront(expressionStatement0);
      TreeSet<String> treeSet0 = new TreeSet<String>();
      SimpleSourceFile simpleSourceFile0 = new SimpleSourceFile("error reporter", false);
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5_STRICT;
      Config config0 = new Config(treeSet0, treeSet0, false, config_LanguageMode0, false);
      // Undeclared exception!
      try { 
        IRFactory.transformTree(astRoot0, simpleSourceFile0, "error reporter", config0, errorCollector0);
        fail("Expecting exception: StackOverflowError");
      
      } catch(StackOverflowError e) {
         //
         // no message in exception (getMessage() returned null)
         //
      }
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      ErrorCollector errorCollector0 = new ErrorCollector();
      Scope scope0 = new Scope(2, 26);
      astRoot0.addChild(scope0);
      LinkedHashSet<String> linkedHashSet0 = new LinkedHashSet<String>();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5;
      Config config0 = new Config(linkedHashSet0, linkedHashSet0, false, config_LanguageMode0, false);
      Node node0 = IRFactory.transformTree(astRoot0, (StaticSourceFile) null, "language version", config0, errorCollector0);
      assertTrue(node0.hasOneChild());
      assertEquals(0, node0.getLength());
      assertTrue(node0.isScript());
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      ErrorCollector errorCollector0 = new ErrorCollector();
      Block block0 = new Block(4, 13);
      astRoot0.addChildToFront(block0);
      TreeSet<String> treeSet0 = new TreeSet<String>();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5_STRICT;
      Config config0 = new Config(treeSet0, treeSet0, false, config_LanguageMode0, true);
      SimpleSourceFile simpleSourceFile0 = new SimpleSourceFile("", false);
      Node node0 = IRFactory.transformTree(astRoot0, simpleSourceFile0, "<Acxw?:`w96f.ithnI", config0, errorCollector0);
      assertTrue(node0.isScript());
      assertEquals(0, node0.getLength());
      assertTrue(node0.hasChildren());
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      Name name0 = new Name();
      FunctionNode functionNode0 = new FunctionNode(4, name0);
      astRoot0.addChildToFront(functionNode0);
      LinkedHashSet<String> linkedHashSet0 = new LinkedHashSet<String>();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5;
      Config config0 = new Config(linkedHashSet0, linkedHashSet0, true, config_LanguageMode0, false);
      ErrorCollector errorCollector0 = new ErrorCollector();
      // Undeclared exception!
      try { 
        IRFactory.transformTree(astRoot0, (StaticSourceFile) null, "error reporter", config0, errorCollector0);
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
      TreeSet<String> treeSet0 = new TreeSet<String>();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5;
      KeywordLiteral keywordLiteral0 = new KeywordLiteral(1, 120);
      astRoot0.addChildToFront(keywordLiteral0);
      Config config0 = new Config(treeSet0, treeSet0, true, config_LanguageMode0, true);
      // Undeclared exception!
      try { 
        IRFactory.transformTree(astRoot0, (StaticSourceFile) null, "nT{M}T:>N\"ZgY}-", config0, (ErrorReporter) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.parsing.IRFactory$TransformDispatcher", e);
      }
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      ErrorCollector errorCollector0 = new ErrorCollector();
      ObjectLiteral objectLiteral0 = new ObjectLiteral(1, 0);
      ThrowStatement throwStatement0 = new ThrowStatement(7, objectLiteral0);
      astRoot0.addChildToFront(throwStatement0);
      TreeSet<String> treeSet0 = new TreeSet<String>();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT3;
      Config config0 = new Config(treeSet0, treeSet0, true, config_LanguageMode0, true);
      SimpleSourceFile simpleSourceFile0 = new SimpleSourceFile("implements", true);
      Node node0 = IRFactory.transformTree(astRoot0, simpleSourceFile0, "@@)X~ZZn~", config0, errorCollector0);
      assertTrue(node0.isScript());
      assertTrue(node0.isFromExterns());
      assertEquals(1, node0.getLength());
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      Token.CommentType token_CommentType0 = Token.CommentType.JSDOC;
      Comment comment0 = new Comment(26, 1, token_CommentType0, "3t72Gm69h/");
      astRoot0.addComment(comment0);
      TreeSet<String> treeSet0 = new TreeSet<String>();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT3;
      Config config0 = new Config(treeSet0, treeSet0, true, config_LanguageMode0, true);
      ToolErrorReporter toolErrorReporter0 = new ToolErrorReporter(true);
      Node node0 = IRFactory.transformTree(astRoot0, (StaticSourceFile) null, "3t72Gm69h/", config0, toolErrorReporter0);
      assertTrue(node0.isScript());
      assertEquals(1, node0.getLength());
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      ErrorCollector errorCollector0 = new ErrorCollector();
      StringLiteral stringLiteral0 = new StringLiteral();
      astRoot0.addChildToFront(stringLiteral0);
      LinkedHashSet<String> linkedHashSet0 = new LinkedHashSet<String>();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5;
      Config config0 = new Config(linkedHashSet0, linkedHashSet0, false, config_LanguageMode0, false);
      SimpleSourceFile simpleSourceFile0 = new SimpleSourceFile("language version", true);
      // Undeclared exception!
      try { 
        IRFactory.transformTree(astRoot0, simpleSourceFile0, "language version", config0, errorCollector0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // StringNode: str is null
         //
         verifyException("com.google.javascript.rhino.Node$StringNode", e);
      }
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      Token.CommentType token_CommentType0 = Token.CommentType.JSDOC;
      Comment comment0 = new Comment(26, 1, token_CommentType0, "3t72Gm69h/");
      astRoot0.addComment(comment0);
      TreeSet<String> treeSet0 = new TreeSet<String>();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5;
      Config config0 = new Config(treeSet0, treeSet0, true, config_LanguageMode0, false);
      SimpleSourceFile simpleSourceFile0 = new SimpleSourceFile("NDE,=}X92uSlC;}r`", false);
      astRoot0.setJsDocNode(comment0);
      ToolErrorReporter toolErrorReporter0 = new ToolErrorReporter(true);
      Node node0 = IRFactory.transformTree(astRoot0, simpleSourceFile0, "rgqnr8K1g:", config0, toolErrorReporter0);
      assertTrue(node0.isScript());
      assertEquals(1, node0.getLength());
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      ErrorCollector errorCollector0 = new ErrorCollector();
      Token.CommentType token_CommentType0 = Token.CommentType.BLOCK_COMMENT;
      Comment comment0 = new Comment(9, 18, token_CommentType0, "getters are not supported in Internet Explorer");
      astRoot0.addComment(comment0);
      TreeSet<String> treeSet0 = new TreeSet<String>();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT3;
      Config config0 = new Config(treeSet0, treeSet0, true, config_LanguageMode0, true);
      Node node0 = IRFactory.transformTree(astRoot0, (StaticSourceFile) null, "getters are not supported in Internet Explorer", config0, errorCollector0);
      assertEquals(1, node0.getLength());
      assertTrue(node0.isScript());
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      ErrorCollector errorCollector0 = new ErrorCollector();
      ReturnStatement returnStatement0 = new ReturnStatement(11);
      astRoot0.addChildToFront(returnStatement0);
      TreeSet<String> treeSet0 = new TreeSet<String>();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT3;
      Config config0 = new Config(treeSet0, treeSet0, false, config_LanguageMode0, true);
      SimpleSourceFile simpleSourceFile0 = new SimpleSourceFile("*m/", true);
      Node node0 = IRFactory.transformTree(astRoot0, simpleSourceFile0, ") {\n", config0, errorCollector0);
      assertTrue(node0.isFromExterns());
      assertEquals(0, node0.getLength());
      assertTrue(node0.isScript());
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      ErrorCollector errorCollector0 = new ErrorCollector();
      ArrayLiteral arrayLiteral0 = new ArrayLiteral();
      astRoot0.addChildToFront(arrayLiteral0);
      LinkedHashSet<String> linkedHashSet0 = new LinkedHashSet<String>();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5;
      Config config0 = new Config(linkedHashSet0, linkedHashSet0, false, config_LanguageMode0, false);
      Node node0 = IRFactory.transformTree(astRoot0, (StaticSourceFile) null, "language version", config0, errorCollector0);
      assertTrue(node0.isScript());
      assertEquals(0, node0.getLength());
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      FunctionNode functionNode0 = new FunctionNode();
      astRoot0.addChildToFront(functionNode0);
      LinkedHashSet<String> linkedHashSet0 = new LinkedHashSet<String>();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5;
      SimpleSourceFile simpleSourceFile0 = new SimpleSourceFile("com.google.javascript.jscomp.ChainableReverseAbstractInterpreter$RestrictByTypeOfResultVisitor", true);
      ToolErrorReporter toolErrorReporter0 = new ToolErrorReporter(false);
      Config config0 = new Config(linkedHashSet0, linkedHashSet0, true, config_LanguageMode0, true);
      Node node0 = IRFactory.transformTree(astRoot0, simpleSourceFile0, "unknown language mode", config0, toolErrorReporter0);
      assertEquals(1, node0.getLength());
      assertEquals(132, node0.getType());
      assertTrue(node0.isFromExterns());
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      ErrorCollector errorCollector0 = new ErrorCollector();
      LabeledStatement labeledStatement0 = new LabeledStatement();
      astRoot0.addChildToFront(labeledStatement0);
      LinkedHashSet<String> linkedHashSet0 = new LinkedHashSet<String>();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5_STRICT;
      Config config0 = new Config(linkedHashSet0, linkedHashSet0, true, config_LanguageMode0, true);
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

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      TreeSet<String> treeSet0 = new TreeSet<String>();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT3;
      Name name0 = new Name();
      astRoot0.addChildToFront(name0);
      Config config0 = new Config(treeSet0, treeSet0, true, config_LanguageMode0, true);
      // Undeclared exception!
      try { 
        IRFactory.transformTree(astRoot0, (StaticSourceFile) null, "Non-JSDoc comment has annotations. Did you mean to start it with '/**'?", config0, (ErrorReporter) null);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // StringNode: str is null
         //
         verifyException("com.google.javascript.rhino.Node$StringNode", e);
      }
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      ErrorCollector errorCollector0 = new ErrorCollector();
      SwitchCase switchCase0 = new SwitchCase();
      astRoot0.addChildToFront(switchCase0);
      LinkedHashSet<String> linkedHashSet0 = new LinkedHashSet<String>();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5_STRICT;
      Config config0 = new Config(linkedHashSet0, linkedHashSet0, false, config_LanguageMode0, false);
      Node node0 = IRFactory.transformTree(astRoot0, (StaticSourceFile) null, "error reporter", config0, errorCollector0);
      assertEquals(0, node0.getLength());
      assertEquals(132, node0.getType());
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      ErrorCollector errorCollector0 = new ErrorCollector();
      VariableDeclaration variableDeclaration0 = new VariableDeclaration();
      astRoot0.addChildToFront(variableDeclaration0);
      TreeSet<String> treeSet0 = new TreeSet<String>();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT3;
      Config config0 = new Config(treeSet0, treeSet0, false, config_LanguageMode0, true);
      Node node0 = IRFactory.transformTree(astRoot0, (StaticSourceFile) null, "language version", config0, errorCollector0);
      assertEquals(0, node0.getLength());
      assertEquals(132, node0.getType());
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      ErrorCollector errorCollector0 = new ErrorCollector();
      VariableDeclaration variableDeclaration0 = new VariableDeclaration();
      astRoot0.addChildToFront(variableDeclaration0);
      TreeSet<String> treeSet0 = new TreeSet<String>();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT3;
      Config config0 = new Config(treeSet0, treeSet0, false, config_LanguageMode0, false);
      Node node0 = IRFactory.transformTree(astRoot0, (StaticSourceFile) null, "language version", config0, errorCollector0);
      assertEquals(0, node0.getLength());
      assertEquals(132, node0.getType());
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      TreeSet<String> treeSet0 = new TreeSet<String>();
      FunctionCall functionCall0 = new FunctionCall(17);
      astRoot0.addChildToFront(functionCall0);
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5;
      Config config0 = new Config(treeSet0, treeSet0, true, config_LanguageMode0, true);
      SimpleSourceFile simpleSourceFile0 = new SimpleSourceFile("language version", true);
      // Undeclared exception!
      try { 
        IRFactory.transformTree(astRoot0, simpleSourceFile0, "error reporter", config0, (ErrorReporter) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
      }
  }
}
