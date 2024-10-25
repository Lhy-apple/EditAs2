/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 21:12:02 GMT 2023
 */

package com.google.javascript.jscomp.parsing;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.google.javascript.jscomp.parsing.Config;
import com.google.javascript.jscomp.parsing.IRFactory;
import com.google.javascript.rhino.JSDocInfo;
import com.google.javascript.rhino.Node;
import com.google.javascript.rhino.head.Token;
import com.google.javascript.rhino.head.ast.ArrayComprehensionLoop;
import com.google.javascript.rhino.head.ast.AstRoot;
import com.google.javascript.rhino.head.ast.Block;
import com.google.javascript.rhino.head.ast.BreakStatement;
import com.google.javascript.rhino.head.ast.Comment;
import com.google.javascript.rhino.head.ast.EmptyExpression;
import com.google.javascript.rhino.head.ast.ErrorCollector;
import com.google.javascript.rhino.head.ast.ExpressionStatement;
import com.google.javascript.rhino.head.ast.FunctionCall;
import com.google.javascript.rhino.head.ast.FunctionNode;
import com.google.javascript.rhino.head.ast.Label;
import com.google.javascript.rhino.head.ast.LabeledStatement;
import com.google.javascript.rhino.head.ast.Name;
import com.google.javascript.rhino.head.ast.NewExpression;
import com.google.javascript.rhino.head.ast.NumberLiteral;
import com.google.javascript.rhino.head.ast.ParenthesizedExpression;
import com.google.javascript.rhino.head.ast.ReturnStatement;
import com.google.javascript.rhino.head.ast.Scope;
import com.google.javascript.rhino.head.ast.SwitchCase;
import com.google.javascript.rhino.head.ast.ThrowStatement;
import com.google.javascript.rhino.head.ast.XmlDotQuery;
import com.google.javascript.rhino.head.tools.ToolErrorReporter;
import com.google.javascript.rhino.jstype.SimpleSourceFile;
import com.google.javascript.rhino.jstype.StaticSourceFile;
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
      ParenthesizedExpression parenthesizedExpression0 = new ParenthesizedExpression(0, (-1));
      astRoot0.addChildrenToFront(parenthesizedExpression0);
      SimpleSourceFile simpleSourceFile0 = new SimpleSourceFile("SHNE", true);
      LinkedHashSet<String> linkedHashSet0 = new LinkedHashSet<String>();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5;
      Config config0 = new Config(linkedHashSet0, linkedHashSet0, true, config_LanguageMode0, true);
      ErrorCollector errorCollector0 = new ErrorCollector();
      // Undeclared exception!
      try { 
        IRFactory.transformTree(astRoot0, simpleSourceFile0, "E9N+9xC%y-J%<`W0V.&", config0, errorCollector0);
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
      SimpleSourceFile simpleSourceFile0 = new SimpleSourceFile("FUNCTION_INSTANCE_TYPE", false);
      LinkedHashSet<String> linkedHashSet0 = new LinkedHashSet<String>();
      EmptyExpression emptyExpression0 = new EmptyExpression();
      astRoot0.addChildToBack(emptyExpression0);
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5_STRICT;
      Config config0 = new Config(linkedHashSet0, linkedHashSet0, false, config_LanguageMode0, false);
      ErrorCollector errorCollector0 = new ErrorCollector();
      Node node0 = IRFactory.transformTree(astRoot0, simpleSourceFile0, "um", config0, errorCollector0);
      assertTrue(node0.hasChildren());
      assertTrue(node0.isScript());
      assertEquals(0, node0.getLength());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      Label label0 = new Label(1);
      astRoot0.addChildToBack(label0);
      SimpleSourceFile simpleSourceFile0 = new SimpleSourceFile("[x~fioRp", true);
      LinkedHashSet<String> linkedHashSet0 = new LinkedHashSet<String>();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT3;
      Config config0 = new Config(linkedHashSet0, linkedHashSet0, true, config_LanguageMode0, true);
      ErrorCollector errorCollector0 = new ErrorCollector();
      // Undeclared exception!
      try { 
        IRFactory.transformTree(astRoot0, simpleSourceFile0, "w/8mQ{3p", config0, errorCollector0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // StringNode: str is null
         //
         verifyException("com.google.javascript.rhino.Node$StringNode", e);
      }
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      NewExpression newExpression0 = new NewExpression();
      astRoot0.addChildToBack(newExpression0);
      SimpleSourceFile simpleSourceFile0 = new SimpleSourceFile("com.google.common.collect.Lists$Partition", false);
      LinkedHashSet<String> linkedHashSet0 = new LinkedHashSet<String>();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5_STRICT;
      Config config0 = new Config(linkedHashSet0, linkedHashSet0, false, config_LanguageMode0, true);
      ErrorCollector errorCollector0 = new ErrorCollector();
      // Undeclared exception!
      try { 
        IRFactory.transformTree(astRoot0, simpleSourceFile0, "com.google.common.collect.Lists$Partition", config0, errorCollector0);
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
      NumberLiteral numberLiteral0 = new NumberLiteral(8);
      astRoot0.addChildToBack(numberLiteral0);
      SimpleSourceFile simpleSourceFile0 = new SimpleSourceFile("PUoc0yj", true);
      LinkedHashSet<String> linkedHashSet0 = new LinkedHashSet<String>();
      ErrorCollector errorCollector0 = new ErrorCollector();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT3;
      Config config0 = new Config(linkedHashSet0, linkedHashSet0, true, config_LanguageMode0, true);
      Node node0 = IRFactory.transformTree(astRoot0, simpleSourceFile0, "PUoc0yj", config0, errorCollector0);
      assertTrue(node0.isFromExterns());
      assertEquals(1, node0.getLength());
      assertEquals(132, node0.getType());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      ExpressionStatement expressionStatement0 = new ExpressionStatement(2, 7);
      astRoot0.addChildToBack(expressionStatement0);
      SimpleSourceFile simpleSourceFile0 = new SimpleSourceFile("BLOCK_COMMENT", true);
      LinkedHashSet<String> linkedHashSet0 = new LinkedHashSet<String>();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT3;
      Config config0 = new Config(linkedHashSet0, linkedHashSet0, true, config_LanguageMode0, true);
      ErrorCollector errorCollector0 = new ErrorCollector();
      // Undeclared exception!
      try { 
        IRFactory.transformTree(astRoot0, simpleSourceFile0, "BLOCK_COMMENT", config0, errorCollector0);
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
      SimpleSourceFile simpleSourceFile0 = new SimpleSourceFile("FUNCTION_INSTANCE_TYPE", false);
      LinkedHashSet<String> linkedHashSet0 = new LinkedHashSet<String>();
      Scope scope0 = new Scope(21);
      astRoot0.addChildToBack(scope0);
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5;
      Config config0 = new Config(linkedHashSet0, linkedHashSet0, false, config_LanguageMode0, true);
      ErrorCollector errorCollector0 = new ErrorCollector();
      Node node0 = IRFactory.transformTree(astRoot0, simpleSourceFile0, "v", config0, errorCollector0);
      assertEquals(0, node0.getLength());
      assertTrue(node0.isScript());
      assertTrue(node0.hasOneChild());
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      Block block0 = new Block();
      astRoot0.addChildToBack(block0);
      SimpleSourceFile simpleSourceFile0 = new SimpleSourceFile("'#9-|c+Gd(,z9-/er", false);
      LinkedHashSet<String> linkedHashSet0 = new LinkedHashSet<String>();
      ErrorCollector errorCollector0 = new ErrorCollector();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT3;
      Config config0 = new Config(linkedHashSet0, linkedHashSet0, false, config_LanguageMode0, false);
      Node node0 = IRFactory.transformTree(astRoot0, simpleSourceFile0, "'#9-|c+Gd(,z9-/er", config0, errorCollector0);
      assertEquals(0, node0.getLength());
      assertEquals(132, node0.getType());
      assertTrue(node0.hasChildren());
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      XmlDotQuery xmlDotQuery0 = new XmlDotQuery(2);
      astRoot0.addChildToBack(xmlDotQuery0);
      SimpleSourceFile simpleSourceFile0 = new SimpleSourceFile("BLOCK_COMMENT", true);
      LinkedHashSet<String> linkedHashSet0 = new LinkedHashSet<String>();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT3;
      Config config0 = new Config(linkedHashSet0, linkedHashSet0, true, config_LanguageMode0, true);
      ErrorCollector errorCollector0 = new ErrorCollector();
      // Undeclared exception!
      try { 
        IRFactory.transformTree(astRoot0, simpleSourceFile0, "BLOCK_COMMENT", config0, errorCollector0);
        fail("Expecting exception: UnsupportedOperationException");
      
      } catch(UnsupportedOperationException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.rhino.head.ast.ErrorCollector", e);
      }
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      ThrowStatement throwStatement0 = new ThrowStatement();
      astRoot0.addChildToBack(throwStatement0);
      SimpleSourceFile simpleSourceFile0 = new SimpleSourceFile("YqO)K|i-vX|})a@%Y3K", false);
      LinkedHashSet<String> linkedHashSet0 = new LinkedHashSet<String>();
      ErrorCollector errorCollector0 = new ErrorCollector();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5;
      Config config0 = new Config(linkedHashSet0, linkedHashSet0, false, config_LanguageMode0, false);
      // Undeclared exception!
      try { 
        IRFactory.transformTree(astRoot0, simpleSourceFile0, "YqO)K|i-vX|})a@%Y3K", config0, errorCollector0);
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
      LinkedHashSet<String> linkedHashSet0 = new LinkedHashSet<String>();
      FunctionNode functionNode0 = new FunctionNode(1);
      astRoot0.addChildToBack(functionNode0);
      SimpleSourceFile simpleSourceFile0 = new SimpleSourceFile(">.1H{<N", false);
      ToolErrorReporter toolErrorReporter0 = new ToolErrorReporter(false);
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5;
      Config config0 = new Config(linkedHashSet0, linkedHashSet0, false, config_LanguageMode0, false);
      Node node0 = IRFactory.transformTree(astRoot0, simpleSourceFile0, ">.1H{<N", config0, toolErrorReporter0);
      assertEquals(0, node0.getLength());
      assertTrue(node0.isScript());
      assertTrue(node0.hasOneChild());
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      LinkedHashSet<String> linkedHashSet0 = new LinkedHashSet<String>();
      Token.CommentType token_CommentType0 = Token.CommentType.JSDOC;
      Comment comment0 = new Comment((-53), 1, token_CommentType0, "use srHct");
      astRoot0.addComment(comment0);
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5_STRICT;
      Config config0 = new Config(linkedHashSet0, (Set<String>) null, true, config_LanguageMode0, true);
      ToolErrorReporter toolErrorReporter0 = new ToolErrorReporter(true);
      SimpleSourceFile simpleSourceFile0 = new SimpleSourceFile("Bad implementaion of call as constructor, name=", true);
      Node node0 = IRFactory.transformTree(astRoot0, simpleSourceFile0, "Bad implementaion of call as constructor, name=", config0, toolErrorReporter0);
      assertEquals(1, node0.getLength());
      assertTrue(node0.isScript());
      assertTrue(node0.isFromExterns());
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      LinkedHashSet<String> linkedHashSet0 = new LinkedHashSet<String>();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT3;
      Config config0 = new Config(linkedHashSet0, linkedHashSet0, true, config_LanguageMode0, true);
      ErrorCollector errorCollector0 = new ErrorCollector();
      Node node0 = IRFactory.transformTree(astRoot0, (StaticSourceFile) null, "BLOCK_COMMENT", config0, errorCollector0);
      assertEquals(1, node0.getLength());
      assertEquals(132, node0.getType());
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      SimpleSourceFile simpleSourceFile0 = new SimpleSourceFile("FUNCTION_INSTANCE_TYPE", false);
      LinkedHashSet<String> linkedHashSet0 = new LinkedHashSet<String>();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5;
      Config config0 = new Config(linkedHashSet0, linkedHashSet0, true, config_LanguageMode0, true);
      ErrorCollector errorCollector0 = new ErrorCollector();
      Token.CommentType token_CommentType0 = Token.CommentType.BLOCK_COMMENT;
      Comment comment0 = new Comment(4, 1734, token_CommentType0, "c$<Q");
      astRoot0.addComment(comment0);
      Node node0 = IRFactory.transformTree(astRoot0, simpleSourceFile0, "c$<Q", config0, errorCollector0);
      assertEquals(1, node0.getLength());
      assertEquals(132, node0.getType());
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      SimpleSourceFile simpleSourceFile0 = new SimpleSourceFile("FUNCTION_INSTANCE_TYPE", false);
      LinkedHashSet<String> linkedHashSet0 = new LinkedHashSet<String>();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5;
      Config config0 = new Config(linkedHashSet0, (Set<String>) null, true, config_LanguageMode0, true);
      ErrorCollector errorCollector0 = new ErrorCollector();
      Token.CommentType token_CommentType0 = Token.CommentType.HTML;
      Comment comment0 = new Comment(0, 67, token_CommentType0, "FUNCTION_INSTANCE_TYPE");
      astRoot0.addComment(comment0);
      Node node0 = IRFactory.transformTree(astRoot0, simpleSourceFile0, "FUNCTION_INSTANCE_TYPE", config0, errorCollector0);
      assertTrue(node0.isScript());
      assertEquals(1, node0.getLength());
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot(51);
      SimpleSourceFile simpleSourceFile0 = new SimpleSourceFile("O&y", true);
      JSDocInfo jSDocInfo0 = new JSDocInfo();
      Set<String> set0 = jSDocInfo0.getModifies();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5;
      Config config0 = new Config(set0, set0, true, config_LanguageMode0, true);
      ToolErrorReporter toolErrorReporter0 = new ToolErrorReporter(false);
      Node node0 = IRFactory.transformTree(astRoot0, simpleSourceFile0, "\n * @", config0, toolErrorReporter0);
      assertTrue(node0.isFromExterns());
      assertEquals(1, node0.getLength());
      assertEquals((-1), node0.getCharno());
      assertEquals(132, node0.getType());
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      BreakStatement breakStatement0 = new BreakStatement((-2665));
      astRoot0.addChildToBack(breakStatement0);
      SimpleSourceFile simpleSourceFile0 = new SimpleSourceFile("8*`Z-BFJR(P", true);
      LinkedHashSet<String> linkedHashSet0 = new LinkedHashSet<String>();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT3;
      Config config0 = new Config(linkedHashSet0, linkedHashSet0, true, config_LanguageMode0, true);
      ErrorCollector errorCollector0 = new ErrorCollector();
      Node node0 = IRFactory.transformTree(astRoot0, simpleSourceFile0, "8*`Z-BFJR(P", config0, errorCollector0);
      assertEquals(1, node0.getLength());
      assertTrue(node0.isFromExterns());
      assertEquals(132, node0.getType());
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      ArrayComprehensionLoop arrayComprehensionLoop0 = new ArrayComprehensionLoop(9, 18);
      astRoot0.addChildToBack(arrayComprehensionLoop0);
      SimpleSourceFile simpleSourceFile0 = new SimpleSourceFile("BLOCK_COMMENT", false);
      LinkedHashSet<String> linkedHashSet0 = new LinkedHashSet<String>();
      ErrorCollector errorCollector0 = new ErrorCollector();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5;
      Config config0 = new Config(linkedHashSet0, linkedHashSet0, false, config_LanguageMode0, false);
      // Undeclared exception!
      try { 
        IRFactory.transformTree(astRoot0, simpleSourceFile0, "BLOCK_COMMENT", config0, errorCollector0);
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
      SimpleSourceFile simpleSourceFile0 = new SimpleSourceFile("FUNCTION_INSTANCE_TYPE", false);
      LinkedHashSet<String> linkedHashSet0 = new LinkedHashSet<String>();
      Name name0 = new Name(1, "FUNCTION_INSTANCE_TYPE");
      FunctionNode functionNode0 = new FunctionNode(20, name0);
      astRoot0.addChildToBack(functionNode0);
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5_STRICT;
      Config config0 = new Config(linkedHashSet0, linkedHashSet0, false, config_LanguageMode0, false);
      ErrorCollector errorCollector0 = new ErrorCollector();
      // Undeclared exception!
      try { 
        IRFactory.transformTree(astRoot0, simpleSourceFile0, "FUNCTION_INSTANCE_TYPE", config0, errorCollector0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
      }
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      LabeledStatement labeledStatement0 = new LabeledStatement();
      astRoot0.addChildToBack(labeledStatement0);
      SimpleSourceFile simpleSourceFile0 = new SimpleSourceFile("2nG~Jqrrm!-Fo", false);
      LinkedHashSet<String> linkedHashSet0 = new LinkedHashSet<String>();
      ErrorCollector errorCollector0 = new ErrorCollector();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5_STRICT;
      Config config0 = new Config(linkedHashSet0, linkedHashSet0, false, config_LanguageMode0, false);
      // Undeclared exception!
      try { 
        IRFactory.transformTree(astRoot0, simpleSourceFile0, "2nG~Jqrrm!-Fo", config0, errorCollector0);
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
      BreakStatement breakStatement0 = new BreakStatement(144);
      Name name0 = new Name(1, 2);
      breakStatement0.setBreakLabel(name0);
      astRoot0.addChildToBack(breakStatement0);
      SimpleSourceFile simpleSourceFile0 = new SimpleSourceFile("FUNCTION_INSTANCE_TYPE", false);
      LinkedHashSet<String> linkedHashSet0 = new LinkedHashSet<String>();
      ErrorCollector errorCollector0 = new ErrorCollector();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT3;
      Config config0 = new Config(linkedHashSet0, linkedHashSet0, true, config_LanguageMode0, true);
      // Undeclared exception!
      try { 
        IRFactory.transformTree(astRoot0, simpleSourceFile0, "FUNCTION_INSTANCE_TYPE", config0, errorCollector0);
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
      AstRoot astRoot0 = new AstRoot();
      ReturnStatement returnStatement0 = new ReturnStatement(9, 23);
      astRoot0.addChildrenToFront(returnStatement0);
      SimpleSourceFile simpleSourceFile0 = new SimpleSourceFile("E9N+9xC%y-J%<`W0V.&", false);
      LinkedHashSet<String> linkedHashSet0 = new LinkedHashSet<String>();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT3;
      Config config0 = new Config(linkedHashSet0, linkedHashSet0, false, config_LanguageMode0, false);
      ErrorCollector errorCollector0 = new ErrorCollector();
      Node node0 = IRFactory.transformTree(astRoot0, simpleSourceFile0, "E9N+9xC%y-J%<`W0V.&", config0, errorCollector0);
      assertEquals(0, node0.getLength());
      assertTrue(node0.hasOneChild());
      assertTrue(node0.isScript());
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      ReturnStatement returnStatement0 = new ReturnStatement(154, 2, astRoot0);
      astRoot0.addChildToBack(returnStatement0);
      SimpleSourceFile simpleSourceFile0 = new SimpleSourceFile("@tS&~&\"hB3zB4-", true);
      LinkedHashSet<String> linkedHashSet0 = new LinkedHashSet<String>();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT3;
      Config config0 = new Config(linkedHashSet0, linkedHashSet0, true, config_LanguageMode0, true);
      ErrorCollector errorCollector0 = new ErrorCollector();
      // Undeclared exception!
      try { 
        IRFactory.transformTree(astRoot0, simpleSourceFile0, "@tS&~&\"hB3zB4-", config0, errorCollector0);
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
      SwitchCase switchCase0 = new SwitchCase();
      astRoot0.addChildToBack(switchCase0);
      SimpleSourceFile simpleSourceFile0 = new SimpleSourceFile("KT", true);
      LinkedHashSet<String> linkedHashSet0 = new LinkedHashSet<String>();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5_STRICT;
      Config config0 = new Config(linkedHashSet0, linkedHashSet0, true, config_LanguageMode0, false);
      ErrorCollector errorCollector0 = new ErrorCollector();
      Node node0 = IRFactory.transformTree(astRoot0, simpleSourceFile0, "KT", config0, errorCollector0);
      assertEquals(1, node0.getLength());
      assertTrue(node0.isFromExterns());
      assertEquals(132, node0.getType());
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      FunctionCall functionCall0 = new FunctionCall();
      astRoot0.addChildToBack(functionCall0);
      SimpleSourceFile simpleSourceFile0 = new SimpleSourceFile("<o#;$bKcFSGxXH@", true);
      LinkedHashSet<String> linkedHashSet0 = new LinkedHashSet<String>();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5_STRICT;
      Config config0 = new Config(linkedHashSet0, linkedHashSet0, true, config_LanguageMode0, false);
      ErrorCollector errorCollector0 = new ErrorCollector();
      // Undeclared exception!
      try { 
        IRFactory.transformTree(astRoot0, simpleSourceFile0, "<o#;$bKcFSGxXH@", config0, errorCollector0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
      }
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      AstRoot astRoot0 = new AstRoot();
      ExpressionStatement expressionStatement0 = new ExpressionStatement();
      expressionStatement0.setHasResult();
      astRoot0.addChildToBack(expressionStatement0);
      SimpleSourceFile simpleSourceFile0 = new SimpleSourceFile("BLOCK_COMMENT", true);
      LinkedHashSet<String> linkedHashSet0 = new LinkedHashSet<String>();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT3;
      Config config0 = new Config(linkedHashSet0, linkedHashSet0, false, config_LanguageMode0, false);
      ErrorCollector errorCollector0 = new ErrorCollector();
      // Undeclared exception!
      try { 
        IRFactory.transformTree(astRoot0, simpleSourceFile0, "BLOCK_COMMENT", config0, errorCollector0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
      }
  }
}
