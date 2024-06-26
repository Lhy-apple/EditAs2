/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 21:23:02 GMT 2023
 */

package com.google.javascript.jscomp.parsing;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.google.javascript.jscomp.parsing.Config;
import com.google.javascript.jscomp.parsing.JsDocInfoParser;
import com.google.javascript.jscomp.parsing.JsDocToken;
import com.google.javascript.jscomp.parsing.JsDocTokenStream;
import com.google.javascript.rhino.JSDocInfo;
import com.google.javascript.rhino.Node;
import com.google.javascript.rhino.head.Context;
import com.google.javascript.rhino.head.ErrorReporter;
import com.google.javascript.rhino.head.Token;
import com.google.javascript.rhino.head.ast.Comment;
import com.google.javascript.rhino.head.ast.ErrorCollector;
import com.google.javascript.rhino.head.tools.ToolErrorReporter;
import com.google.javascript.rhino.jstype.SimpleSourceFile;
import java.util.HashSet;
import java.util.LinkedHashSet;
import java.util.Set;
import java.util.TreeSet;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class JsDocInfoParser_ESTest extends JsDocInfoParser_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      JsDocTokenStream jsDocTokenStream0 = new JsDocTokenStream("2%*/Db#j[yK", 55);
      Token.CommentType token_CommentType0 = Token.CommentType.HTML;
      Comment comment0 = new Comment(20, 109, token_CommentType0, "");
      TreeSet<String> treeSet0 = new TreeSet<String>();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT3;
      Config config0 = new Config(treeSet0, treeSet0, false, config_LanguageMode0, true);
      Context context0 = Context.getCurrentContext();
      ErrorReporter errorReporter0 = context0.getErrorReporter();
      JsDocInfoParser jsDocInfoParser0 = new JsDocInfoParser(jsDocTokenStream0, comment0, (Node) null, config0, errorReporter0);
      jsDocInfoParser0.parse();
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      JsDocTokenStream jsDocTokenStream0 = new JsDocTokenStream("\"y]1Dg+3");
      Node node0 = Node.newString("\"y]1Dg+3");
      JSDocInfo jSDocInfo0 = new JSDocInfo();
      Set<String> set0 = jSDocInfo0.getModifies();
      Context context0 = Context.enter();
      ErrorReporter errorReporter0 = context0.getErrorReporter();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5_STRICT;
      Config config0 = new Config(set0, set0, true, config_LanguageMode0, false);
      JsDocInfoParser jsDocInfoParser0 = new JsDocInfoParser(jsDocTokenStream0, (Comment) null, node0, config0, errorReporter0);
      JSDocInfo jSDocInfo1 = jsDocInfoParser0.parseInlineTypeDoc();
      assertNotNull(jSDocInfo1);
      
      JsDocToken jsDocToken0 = JsDocToken.QMARK;
      Node node1 = jsDocInfoParser0.parseAndRecordTypeNode(jsDocToken0);
      assertNotNull(node1);
      assertEquals(304, node1.getType());
      assertFalse(node1.hasChildren());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      ErrorCollector errorCollector0 = new ErrorCollector();
      JsDocTokenStream jsDocTokenStream0 = new JsDocTokenStream("*J&OZ", 9, 150);
      Token.CommentType token_CommentType0 = Token.CommentType.BLOCK_COMMENT;
      Comment comment0 = new Comment((-1), 160, token_CommentType0, "language version");
      Node node0 = new Node(1, 6, 1);
      HashSet<String> hashSet0 = new HashSet<String>();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5;
      Config config0 = new Config(hashSet0, hashSet0, false, config_LanguageMode0, false);
      JsDocInfoParser jsDocInfoParser0 = new JsDocInfoParser(jsDocTokenStream0, comment0, node0, config0, errorCollector0);
      boolean boolean0 = jsDocInfoParser0.hasParsedJSDocInfo();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      JsDocTokenStream jsDocTokenStream0 = new JsDocTokenStream("\"y]1ag+3");
      Node node0 = Node.newString("\"y]1ag+3");
      JSDocInfo jSDocInfo0 = new JSDocInfo();
      Set<String> set0 = jSDocInfo0.getModifies();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5_STRICT;
      Config config0 = new Config(set0, set0, true, config_LanguageMode0, true);
      ToolErrorReporter toolErrorReporter0 = new ToolErrorReporter(true);
      JsDocInfoParser jsDocInfoParser0 = new JsDocInfoParser(jsDocTokenStream0, (Comment) null, node0, config0, toolErrorReporter0);
      JSDocInfo jSDocInfo1 = jsDocInfoParser0.getFileOverviewJSDocInfo();
      assertNull(jSDocInfo1);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      JsDocTokenStream jsDocTokenStream0 = new JsDocTokenStream("y]1g+3");
      Node node0 = JsDocInfoParser.parseTypeString("y]1g+3");
      assertNotNull(node0);
      
      LinkedHashSet<String> linkedHashSet0 = new LinkedHashSet<String>();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT3;
      Config config0 = new Config(linkedHashSet0, linkedHashSet0, false, config_LanguageMode0, false);
      ToolErrorReporter toolErrorReporter0 = new ToolErrorReporter(false);
      JsDocInfoParser jsDocInfoParser0 = new JsDocInfoParser(jsDocTokenStream0, (Comment) null, node0, config0, toolErrorReporter0);
      Node.FileLevelJsDocBuilder node_FileLevelJsDocBuilder0 = node0.new FileLevelJsDocBuilder();
      jsDocInfoParser0.setFileLevelJsDocBuilder(node_FileLevelJsDocBuilder0);
      assertEquals(40, node0.getType());
      assertEquals(0, node0.getSourcePosition());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      JsDocTokenStream jsDocTokenStream0 = new JsDocTokenStream("{o8zF$N(@&");
      Node node0 = Node.newString("{o8zF$N(@&");
      JSDocInfo jSDocInfo0 = new JSDocInfo();
      Set<String> set0 = jSDocInfo0.getSuppressions();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT3;
      Config config0 = new Config(set0, set0, true, config_LanguageMode0, true);
      ToolErrorReporter toolErrorReporter0 = new ToolErrorReporter(true);
      JsDocInfoParser jsDocInfoParser0 = new JsDocInfoParser(jsDocTokenStream0, (Comment) null, node0, config0, toolErrorReporter0);
      jsDocInfoParser0.setFileOverviewJSDocInfo(jSDocInfo0);
      assertNull(jSDocInfo0.getLicense());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Node node0 = JsDocInfoParser.parseTypeString("(Ljava/lang/Object;)D");
      assertEquals(301, node0.getType());
      assertNotNull(node0);
      assertTrue(node0.hasOneChild());
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Node node0 = JsDocInfoParser.parseTypeString("{o8zF$N(@&");
      assertNull(node0);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      JsDocTokenStream jsDocTokenStream0 = new JsDocTokenStream("\"y]1ag+3");
      Node node0 = JsDocInfoParser.parseTypeString("\"y]1ag+3");
      SimpleSourceFile simpleSourceFile0 = new SimpleSourceFile("HT", true);
      node0.setStaticSourceFile(simpleSourceFile0);
      JSDocInfo jSDocInfo0 = new JSDocInfo();
      Set<String> set0 = jSDocInfo0.getModifies();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT3;
      Config config0 = new Config(set0, set0, true, config_LanguageMode0, true);
      ToolErrorReporter toolErrorReporter0 = new ToolErrorReporter(false);
      JsDocInfoParser jsDocInfoParser0 = new JsDocInfoParser(jsDocTokenStream0, (Comment) null, node0, config0, toolErrorReporter0);
      JsDocToken jsDocToken0 = JsDocToken.ANNOTATION;
      Node node1 = jsDocInfoParser0.parseAndRecordTypeNode(jsDocToken0);
      assertEquals(0, node0.getSourcePosition());
      assertNull(node1);
      assertEquals(0, node0.getLineno());
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      JsDocTokenStream jsDocTokenStream0 = new JsDocTokenStream("function");
      Node node0 = JsDocInfoParser.parseTypeString("J*&~ftV");
      HashSet<String> hashSet0 = new HashSet<String>();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5_STRICT;
      Config config0 = new Config(hashSet0, hashSet0, false, config_LanguageMode0, false);
      ToolErrorReporter toolErrorReporter0 = new ToolErrorReporter(false);
      JsDocInfoParser jsDocInfoParser0 = new JsDocInfoParser(jsDocTokenStream0, (Comment) null, node0, config0, toolErrorReporter0);
      JSDocInfo jSDocInfo0 = jsDocInfoParser0.parseInlineTypeDoc();
      assertNull(jSDocInfo0);
      assertEquals(0, node0.getSourcePosition());
      assertEquals(0, node0.getLineno());
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      JsDocTokenStream jsDocTokenStream0 = new JsDocTokenStream("{o8z$(@&");
      HashSet<String> hashSet0 = new HashSet<String>();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5_STRICT;
      Config config0 = new Config(hashSet0, hashSet0, false, config_LanguageMode0, false);
      ToolErrorReporter toolErrorReporter0 = new ToolErrorReporter(false);
      JsDocInfoParser jsDocInfoParser0 = new JsDocInfoParser(jsDocTokenStream0, (Comment) null, (Node) null, config0, toolErrorReporter0);
      JsDocToken jsDocToken0 = JsDocToken.ANNOTATION;
      jsDocInfoParser0.parseAndRecordTypeNode(jsDocToken0);
      boolean boolean0 = jsDocInfoParser0.parse();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      JsDocTokenStream jsDocTokenStream0 = new JsDocTokenStream("y]1g+3");
      Node node0 = JsDocInfoParser.parseTypeString("y]1g+3");
      JSDocInfo jSDocInfo0 = new JSDocInfo();
      Set<String> set0 = jSDocInfo0.getModifies();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT3;
      Config config0 = new Config(set0, set0, false, config_LanguageMode0, false);
      ToolErrorReporter toolErrorReporter0 = new ToolErrorReporter(false);
      JsDocInfoParser jsDocInfoParser0 = new JsDocInfoParser(jsDocTokenStream0, (Comment) null, node0, config0, toolErrorReporter0);
      JsDocToken jsDocToken0 = JsDocToken.EOC;
      jsDocInfoParser0.parseAndRecordTypeNode(jsDocToken0);
      boolean boolean0 = jsDocInfoParser0.parse();
      assertEquals(0, node0.getSourcePosition());
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      Context context0 = new Context();
      JsDocTokenStream jsDocTokenStream0 = new JsDocTokenStream("*J&OZ");
      Token.CommentType token_CommentType0 = Token.CommentType.BLOCK_COMMENT;
      Comment comment0 = new Comment((-1), 160, token_CommentType0, "language version");
      LinkedHashSet<String> linkedHashSet0 = new LinkedHashSet<String>();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5_STRICT;
      Config config0 = new Config(linkedHashSet0, linkedHashSet0, true, config_LanguageMode0, true);
      ErrorReporter errorReporter0 = context0.getErrorReporter();
      JsDocInfoParser jsDocInfoParser0 = new JsDocInfoParser(jsDocTokenStream0, comment0, (Node) null, config0, errorReporter0);
      JsDocToken jsDocToken0 = JsDocToken.ANNOTATION;
      jsDocInfoParser0.parseAndRecordTypeNode(jsDocToken0);
      boolean boolean0 = jsDocInfoParser0.parse();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      ErrorCollector errorCollector0 = new ErrorCollector();
      JsDocTokenStream jsDocTokenStream0 = new JsDocTokenStream("*J&OZ", 9, 150);
      Token.CommentType token_CommentType0 = Token.CommentType.BLOCK_COMMENT;
      Comment comment0 = new Comment((-1), 160, token_CommentType0, "language version");
      Node node0 = Node.newString(17, "language version");
      HashSet<String> hashSet0 = new HashSet<String>();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5;
      Config config0 = new Config(hashSet0, hashSet0, false, config_LanguageMode0, false);
      JsDocInfoParser jsDocInfoParser0 = new JsDocInfoParser(jsDocTokenStream0, comment0, node0, config0, errorCollector0);
      // Undeclared exception!
      try { 
        jsDocInfoParser0.parse();
        fail("Expecting exception: UnsupportedOperationException");
      
      } catch(UnsupportedOperationException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.rhino.head.ast.ErrorCollector", e);
      }
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      JsDocTokenStream jsDocTokenStream0 = new JsDocTokenStream("{o8zF$N(@&");
      JSDocInfo jSDocInfo0 = new JSDocInfo();
      Set<String> set0 = jSDocInfo0.getModifies();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT3;
      Config config0 = new Config(set0, set0, true, config_LanguageMode0, true);
      ToolErrorReporter toolErrorReporter0 = new ToolErrorReporter(true);
      JsDocInfoParser jsDocInfoParser0 = new JsDocInfoParser(jsDocTokenStream0, (Comment) null, (Node) null, config0, toolErrorReporter0);
      jsDocInfoParser0.parseInlineTypeDoc();
      boolean boolean0 = jsDocInfoParser0.parse();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      JsDocTokenStream jsDocTokenStream0 = new JsDocTokenStream("!", 1073741789);
      Token.CommentType token_CommentType0 = Token.CommentType.BLOCK_COMMENT;
      Comment comment0 = new Comment(1073741789, 1073741789, token_CommentType0, "!");
      Node node0 = Node.newString(10, "!");
      JSDocInfo jSDocInfo0 = new JSDocInfo();
      Set<String> set0 = jSDocInfo0.getModifies();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5;
      Config config0 = new Config(set0, set0, true, config_LanguageMode0, true);
      ErrorCollector errorCollector0 = new ErrorCollector();
      JsDocInfoParser jsDocInfoParser0 = new JsDocInfoParser(jsDocTokenStream0, comment0, node0, config0, errorCollector0);
      // Undeclared exception!
      try { 
        jsDocInfoParser0.parse();
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
      JsDocTokenStream jsDocTokenStream0 = new JsDocTokenStream("y]1g+3");
      Node node0 = JsDocInfoParser.parseTypeString("y]1g+3");
      JSDocInfo jSDocInfo0 = new JSDocInfo();
      Set<String> set0 = jSDocInfo0.getModifies();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT3;
      Config config0 = new Config(set0, set0, true, config_LanguageMode0, true);
      ToolErrorReporter toolErrorReporter0 = new ToolErrorReporter(true);
      JsDocInfoParser jsDocInfoParser0 = new JsDocInfoParser(jsDocTokenStream0, (Comment) null, node0, config0, toolErrorReporter0);
      JsDocToken jsDocToken0 = JsDocToken.COMMA;
      Node node1 = jsDocInfoParser0.parseAndRecordTypeNode(jsDocToken0);
      assertNull(node1);
      
      jsDocInfoParser0.parse();
      assertEquals(0, node0.getSourcePosition());
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      JsDocTokenStream jsDocTokenStream0 = new JsDocTokenStream("{o8zF$N(@&");
      JSDocInfo jSDocInfo0 = new JSDocInfo();
      Set<String> set0 = jSDocInfo0.getModifies();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT3;
      Config config0 = new Config(set0, set0, true, config_LanguageMode0, true);
      ToolErrorReporter toolErrorReporter0 = new ToolErrorReporter(true);
      JsDocInfoParser jsDocInfoParser0 = new JsDocInfoParser(jsDocTokenStream0, (Comment) null, (Node) null, config0, toolErrorReporter0);
      JsDocToken jsDocToken0 = JsDocToken.COLON;
      jsDocInfoParser0.parseAndRecordTypeNode(jsDocToken0);
      boolean boolean0 = jsDocInfoParser0.parse();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      Node node0 = Node.newString("\"y]1Dg+3");
      JSDocInfo jSDocInfo0 = new JSDocInfo();
      Set<String> set0 = jSDocInfo0.getModifies();
      JsDocTokenStream jsDocTokenStream0 = new JsDocTokenStream("n^>kR:");
      Context context0 = Context.enter();
      ErrorReporter errorReporter0 = context0.getErrorReporter();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5_STRICT;
      Config config0 = new Config(set0, set0, true, config_LanguageMode0, false);
      JsDocInfoParser jsDocInfoParser0 = new JsDocInfoParser(jsDocTokenStream0, (Comment) null, node0, config0, errorReporter0);
      jsDocInfoParser0.parseInlineTypeDoc();
      boolean boolean0 = jsDocInfoParser0.parse();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      JsDocTokenStream jsDocTokenStream0 = new JsDocTokenStream("[qNZ@>=[XIo.+;f@'=");
      TreeSet<String> treeSet0 = new TreeSet<String>();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5;
      Config config0 = new Config(treeSet0, treeSet0, true, config_LanguageMode0, true);
      ErrorCollector errorCollector0 = new ErrorCollector();
      JsDocInfoParser jsDocInfoParser0 = new JsDocInfoParser(jsDocTokenStream0, (Comment) null, (Node) null, config0, errorCollector0);
      // Undeclared exception!
      try { 
        jsDocInfoParser0.parse();
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
      JsDocTokenStream jsDocTokenStream0 = new JsDocTokenStream("{o8z$(h@&");
      JSDocInfo jSDocInfo0 = new JSDocInfo();
      Set<String> set0 = jSDocInfo0.getModifies();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5;
      Config config0 = new Config(set0, set0, true, config_LanguageMode0, true);
      ToolErrorReporter toolErrorReporter0 = new ToolErrorReporter(true);
      JsDocInfoParser jsDocInfoParser0 = new JsDocInfoParser(jsDocTokenStream0, (Comment) null, (Node) null, config0, toolErrorReporter0);
      boolean boolean0 = jsDocInfoParser0.parse();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      JsDocTokenStream jsDocTokenStream0 = new JsDocTokenStream("?y]1g+3");
      Node node0 = JsDocInfoParser.parseTypeString("?y]1g+3");
      JSDocInfo jSDocInfo0 = new JSDocInfo();
      Set<String> set0 = jSDocInfo0.getModifies();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5;
      Config config0 = new Config(set0, set0, true, config_LanguageMode0, true);
      ToolErrorReporter toolErrorReporter0 = new ToolErrorReporter(true);
      JsDocInfoParser jsDocInfoParser0 = new JsDocInfoParser(jsDocTokenStream0, (Comment) null, node0, config0, toolErrorReporter0);
      JsDocToken jsDocToken0 = JsDocToken.LT;
      Node node1 = jsDocInfoParser0.parseAndRecordTypeNode(jsDocToken0);
      assertNull(node1);
      
      jsDocInfoParser0.parse();
      assertEquals(2, node0.getSourcePosition());
      assertEquals(304, node0.getType());
      assertEquals(2, node0.getCharno());
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      JsDocTokenStream jsDocTokenStream0 = new JsDocTokenStream("?y]1g+3");
      Node node0 = JsDocInfoParser.parseTypeString("?y]1g+3");
      assertNotNull(node0);
      
      JSDocInfo jSDocInfo0 = new JSDocInfo();
      Set<String> set0 = jSDocInfo0.getModifies();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5;
      Config config0 = new Config(set0, set0, true, config_LanguageMode0, true);
      ToolErrorReporter toolErrorReporter0 = new ToolErrorReporter(true);
      JsDocInfoParser jsDocInfoParser0 = new JsDocInfoParser(jsDocTokenStream0, (Comment) null, node0, config0, toolErrorReporter0);
      jsDocInfoParser0.parse();
      assertEquals(2, node0.getCharno());
      assertEquals(304, node0.getType());
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      JsDocTokenStream jsDocTokenStream0 = new JsDocTokenStream("\"y]1Dg+3");
      Node node0 = Node.newString("\"y]1Dg+3");
      HashSet<String> hashSet0 = new HashSet<String>();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5_STRICT;
      Config config0 = new Config(hashSet0, hashSet0, true, config_LanguageMode0, true);
      ToolErrorReporter toolErrorReporter0 = new ToolErrorReporter(true);
      JsDocInfoParser jsDocInfoParser0 = new JsDocInfoParser(jsDocTokenStream0, (Comment) null, node0, config0, toolErrorReporter0);
      JsDocToken jsDocToken0 = JsDocToken.PIPE;
      jsDocInfoParser0.parseAndRecordTypeNode(jsDocToken0);
      boolean boolean0 = jsDocInfoParser0.parse();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      JsDocTokenStream jsDocTokenStream0 = new JsDocTokenStream("\"y]1Dg+3");
      Node node0 = Node.newString("\"y]1Dg+3");
      JSDocInfo jSDocInfo0 = new JSDocInfo();
      Set<String> set0 = jSDocInfo0.getModifies();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5_STRICT;
      Config config0 = new Config(set0, set0, true, config_LanguageMode0, true);
      ToolErrorReporter toolErrorReporter0 = new ToolErrorReporter(true);
      JsDocInfoParser jsDocInfoParser0 = new JsDocInfoParser(jsDocTokenStream0, (Comment) null, node0, config0, toolErrorReporter0);
      jsDocInfoParser0.parseInlineTypeDoc();
      boolean boolean0 = jsDocInfoParser0.parse();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      JsDocTokenStream jsDocTokenStream0 = new JsDocTokenStream("y]1g+3");
      Node node0 = JsDocInfoParser.parseTypeString("y]1g+3");
      JSDocInfo jSDocInfo0 = new JSDocInfo();
      Set<String> set0 = jSDocInfo0.getModifies();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT3;
      Config config0 = new Config(set0, set0, true, config_LanguageMode0, true);
      ToolErrorReporter toolErrorReporter0 = new ToolErrorReporter(true);
      JsDocInfoParser jsDocInfoParser0 = new JsDocInfoParser(jsDocTokenStream0, (Comment) null, node0, config0, toolErrorReporter0);
      JsDocToken jsDocToken0 = JsDocToken.RC;
      Node node1 = jsDocInfoParser0.parseAndRecordTypeNode(jsDocToken0);
      assertNull(node1);
      
      jsDocInfoParser0.parse();
      assertEquals(0, node0.getSourcePosition());
      assertEquals(0, node0.getLineno());
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      JsDocTokenStream jsDocTokenStream0 = new JsDocTokenStream("y]1g+3");
      Node node0 = JsDocInfoParser.parseTypeString("y]1g+3");
      JSDocInfo jSDocInfo0 = new JSDocInfo();
      Set<String> set0 = jSDocInfo0.getModifies();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT3;
      Config config0 = new Config(set0, set0, true, config_LanguageMode0, true);
      ToolErrorReporter toolErrorReporter0 = new ToolErrorReporter(true);
      JsDocInfoParser jsDocInfoParser0 = new JsDocInfoParser(jsDocTokenStream0, (Comment) null, node0, config0, toolErrorReporter0);
      JsDocToken jsDocToken0 = JsDocToken.RP;
      Node node1 = jsDocInfoParser0.parseAndRecordTypeNode(jsDocToken0);
      assertNull(node1);
      
      jsDocInfoParser0.parse();
      assertEquals(0, node0.getSourcePosition());
      assertTrue(node0.isString());
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      JsDocTokenStream jsDocTokenStream0 = new JsDocTokenStream("?y]1g+3");
      Node node0 = JsDocInfoParser.parseTypeString("?y]1g+3");
      assertNotNull(node0);
      
      JSDocInfo jSDocInfo0 = new JSDocInfo();
      Set<String> set0 = jSDocInfo0.getModifies();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5_STRICT;
      Config config0 = new Config(set0, set0, true, config_LanguageMode0, true);
      ToolErrorReporter toolErrorReporter0 = new ToolErrorReporter(true);
      JsDocInfoParser jsDocInfoParser0 = new JsDocInfoParser(jsDocTokenStream0, (Comment) null, node0, config0, toolErrorReporter0);
      JsDocToken jsDocToken0 = JsDocToken.ELLIPSIS;
      Node node1 = jsDocInfoParser0.parseAndRecordTypeNode(jsDocToken0);
      assertNull(node1);
      
      jsDocInfoParser0.parse();
      assertEquals(304, node0.getType());
      assertEquals(2, node0.getCharno());
      assertTrue(node0.hasChildren());
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      JsDocTokenStream jsDocTokenStream0 = new JsDocTokenStream("y]1g+3");
      Node node0 = JsDocInfoParser.parseTypeString("y]1g+3");
      JSDocInfo jSDocInfo0 = new JSDocInfo();
      Set<String> set0 = jSDocInfo0.getModifies();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5;
      Config config0 = new Config(set0, set0, true, config_LanguageMode0, true);
      ToolErrorReporter toolErrorReporter0 = new ToolErrorReporter(true);
      JsDocInfoParser jsDocInfoParser0 = new JsDocInfoParser(jsDocTokenStream0, (Comment) null, node0, config0, toolErrorReporter0);
      JsDocToken jsDocToken0 = JsDocToken.EQUALS;
      Node node1 = jsDocInfoParser0.parseAndRecordTypeNode(jsDocToken0);
      assertNull(node1);
      
      jsDocInfoParser0.parse();
      assertEquals(0, node0.getSourcePosition());
      assertEquals(0, node0.getCharno());
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      JsDocTokenStream jsDocTokenStream0 = new JsDocTokenStream("\"y]1Dg+3");
      Node node0 = Node.newString("\"y]1Dg+3");
      JSDocInfo jSDocInfo0 = new JSDocInfo();
      Set<String> set0 = jSDocInfo0.getModifies();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5_STRICT;
      Config config0 = new Config(set0, set0, true, config_LanguageMode0, true);
      ToolErrorReporter toolErrorReporter0 = new ToolErrorReporter(true);
      JsDocInfoParser jsDocInfoParser0 = new JsDocInfoParser(jsDocTokenStream0, (Comment) null, node0, config0, toolErrorReporter0);
      JsDocToken jsDocToken0 = JsDocToken.EOC;
      jsDocInfoParser0.parseAndRecordTypeNode(jsDocToken0);
      boolean boolean0 = jsDocInfoParser0.parse();
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      JsDocTokenStream jsDocTokenStream0 = new JsDocTokenStream("\n\nTree2:\n", (-1));
      Node node0 = Node.newString((-7), "", 16, 1737);
      JSDocInfo jSDocInfo0 = new JSDocInfo();
      Set<String> set0 = jSDocInfo0.getModifies();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT3;
      Config config0 = new Config(set0, set0, true, config_LanguageMode0, false);
      ErrorCollector errorCollector0 = new ErrorCollector();
      JsDocInfoParser jsDocInfoParser0 = new JsDocInfoParser(jsDocTokenStream0, (Comment) null, node0, config0, errorCollector0);
      // Undeclared exception!
      try { 
        jsDocInfoParser0.parse();
        fail("Expecting exception: UnsupportedOperationException");
      
      } catch(UnsupportedOperationException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.rhino.head.ast.ErrorCollector", e);
      }
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      ErrorCollector errorCollector0 = new ErrorCollector();
      JsDocTokenStream jsDocTokenStream0 = new JsDocTokenStream("*J&OZ", 9, 150);
      Token.CommentType token_CommentType0 = Token.CommentType.BLOCK_COMMENT;
      Comment comment0 = new Comment((-1), 160, token_CommentType0, "language version");
      LinkedHashSet<String> linkedHashSet0 = new LinkedHashSet<String>();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5_STRICT;
      Config config0 = new Config(linkedHashSet0, linkedHashSet0, true, config_LanguageMode0, true);
      JsDocInfoParser jsDocInfoParser0 = new JsDocInfoParser(jsDocTokenStream0, comment0, (Node) null, config0, errorCollector0);
      // Undeclared exception!
      try { 
        jsDocInfoParser0.parse();
        fail("Expecting exception: UnsupportedOperationException");
      
      } catch(UnsupportedOperationException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.rhino.head.ast.ErrorCollector", e);
      }
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      Node node0 = JsDocInfoParser.parseTypeString("y]1g+3");
      JSDocInfo jSDocInfo0 = new JSDocInfo();
      Set<String> set0 = jSDocInfo0.getModifies();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT3;
      Config config0 = new Config(set0, set0, true, config_LanguageMode0, true);
      ToolErrorReporter toolErrorReporter0 = new ToolErrorReporter(true);
      JsDocTokenStream jsDocTokenStream0 = new JsDocTokenStream(" <= ");
      JsDocInfoParser jsDocInfoParser0 = new JsDocInfoParser(jsDocTokenStream0, (Comment) null, node0, config0, toolErrorReporter0);
      jsDocInfoParser0.parse();
      assertEquals(1, node0.getLength());
      assertEquals(0, node0.getSourcePosition());
      assertEquals(0, node0.getLineno());
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      JsDocTokenStream jsDocTokenStream0 = new JsDocTokenStream("@y]Vag+3");
      Node node0 = Node.newString("@y]Vag+3");
      JSDocInfo jSDocInfo0 = new JSDocInfo();
      Set<String> set0 = jSDocInfo0.getModifies();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT3;
      Config config0 = new Config(set0, set0, true, config_LanguageMode0, true);
      ToolErrorReporter toolErrorReporter0 = new ToolErrorReporter(true);
      JsDocInfoParser jsDocInfoParser0 = new JsDocInfoParser(jsDocTokenStream0, (Comment) null, node0, config0, toolErrorReporter0);
      JsDocToken jsDocToken0 = JsDocToken.LC;
      Node node1 = jsDocInfoParser0.parseAndRecordTypeNode(jsDocToken0);
      assertNull(node1);
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      Node node0 = JsDocInfoParser.parseTypeString("8^IKi;##Wb|sR,ow@>^");
      assertEquals(3, node0.getChildCount());
      assertNotNull(node0);
      assertEquals(301, node0.getType());
  }

  @Test(timeout = 4000)
  public void test35()  throws Throwable  {
      Node node0 = Node.newString("x`t;s,'8VHR>F]");
      JSDocInfo jSDocInfo0 = new JSDocInfo();
      Set<String> set0 = jSDocInfo0.getModifies();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5;
      LinkedHashSet<String> linkedHashSet0 = new LinkedHashSet<String>();
      Config config0 = new Config(set0, linkedHashSet0, false, config_LanguageMode0, true);
      ToolErrorReporter toolErrorReporter0 = new ToolErrorReporter(false);
      JsDocTokenStream jsDocTokenStream0 = new JsDocTokenStream("x`t;s,'8VHR>F]");
      JsDocInfoParser jsDocInfoParser0 = new JsDocInfoParser(jsDocTokenStream0, (Comment) null, node0, config0, toolErrorReporter0);
      JSDocInfo jSDocInfo1 = jsDocInfoParser0.parseInlineTypeDoc();
      assertNotNull(jSDocInfo1);
      
      JsDocToken jsDocToken0 = JsDocToken.QMARK;
      Node node1 = jsDocInfoParser0.parseAndRecordTypeNode(jsDocToken0);
      assertFalse(node1.hasOneChild());
      assertEquals(304, node1.getType());
      assertNotNull(node1);
  }

  @Test(timeout = 4000)
  public void test36()  throws Throwable  {
      Node node0 = JsDocInfoParser.parseTypeString("&|?}IL 7&V");
      assertNotNull(node0);
      assertEquals(301, node0.getType());
      assertFalse(node0.hasOneChild());
  }

  @Test(timeout = 4000)
  public void test37()  throws Throwable  {
      Node node0 = Node.newString("\"y]1Dg+3");
      JSDocInfo jSDocInfo0 = new JSDocInfo();
      Set<String> set0 = jSDocInfo0.getModifies();
      JsDocTokenStream jsDocTokenStream0 = new JsDocTokenStream("n^>kR:");
      Context context0 = Context.enter();
      ErrorReporter errorReporter0 = context0.getErrorReporter();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5_STRICT;
      Config config0 = new Config(set0, set0, true, config_LanguageMode0, false);
      JsDocInfoParser jsDocInfoParser0 = new JsDocInfoParser(jsDocTokenStream0, (Comment) null, node0, config0, errorReporter0);
      JSDocInfo jSDocInfo1 = jsDocInfoParser0.parseInlineTypeDoc();
      assertNotNull(jSDocInfo1);
      
      JsDocToken jsDocToken0 = JsDocToken.QMARK;
      Node node1 = jsDocInfoParser0.parseAndRecordTypeNode(jsDocToken0);
      assertEquals(304, node1.getType());
      assertNotNull(node1);
      assertEquals(2, node1.getCharno());
  }

  @Test(timeout = 4000)
  public void test38()  throws Throwable  {
      Node node0 = JsDocInfoParser.parseTypeString("o2?");
      assertNotNull(node0);
      assertEquals(304, node0.getType());
  }

  @Test(timeout = 4000)
  public void test39()  throws Throwable  {
      Node node0 = JsDocInfoParser.parseTypeString("X25!/v`W`x3Y?W.");
      assertNotNull(node0);
      assertEquals(306, node0.getType());
  }

  @Test(timeout = 4000)
  public void test40()  throws Throwable  {
      Node node0 = JsDocInfoParser.parseTypeString("!*c~u]ZXuJm0y/MAf=>");
      assertEquals(306, node0.getType());
      assertNotNull(node0);
  }

  @Test(timeout = 4000)
  public void test41()  throws Throwable  {
      Node node0 = JsDocInfoParser.parseTypeString("[qNZ@>=[XIo.+;f@'=");
      assertNull(node0);
  }

  @Test(timeout = 4000)
  public void test42()  throws Throwable  {
      Node node0 = JsDocInfoParser.parseTypeString("null");
      assertNotNull(node0);
      assertEquals(0, node0.getSourcePosition());
      assertEquals(40, node0.getType());
  }

  @Test(timeout = 4000)
  public void test43()  throws Throwable  {
      Node node0 = JsDocInfoParser.parseTypeString("undefined");
      assertNotNull(node0);
      assertEquals(0, node0.getSourcePosition());
      assertTrue(node0.isString());
  }

  @Test(timeout = 4000)
  public void test44()  throws Throwable  {
      Node node0 = JsDocInfoParser.parseTypeString("w|_fv;C|8<<=c2BsB");
      assertEquals(3, node0.getChildCount());
      assertEquals(301, node0.getType());
      assertNotNull(node0);
  }

  @Test(timeout = 4000)
  public void test45()  throws Throwable  {
      Node node0 = JsDocInfoParser.parseTypeString("{o8zF$N4:(@&");
      assertNull(node0);
  }

  @Test(timeout = 4000)
  public void test46()  throws Throwable  {
      Node node0 = JsDocInfoParser.parseTypeString("(Ljava/lang/Object;Lorg/mozilla/javascript/Context;Lorg/mozilla/javascript/Scriptable;[Ljava/lang/Object;)Lorg/mozilla/javascript/Scriptable;");
      assertNull(node0);
  }

  @Test(timeout = 4000)
  public void test47()  throws Throwable  {
      Node node0 = JsDocInfoParser.parseTypeString("h");
      assertNotNull(node0);
      
      JSDocInfo jSDocInfo0 = new JSDocInfo();
      Set<String> set0 = jSDocInfo0.getModifies();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5;
      Config config0 = new Config(set0, set0, false, config_LanguageMode0, true);
      ToolErrorReporter toolErrorReporter0 = new ToolErrorReporter(true);
      JsDocTokenStream jsDocTokenStream0 = new JsDocTokenStream("...", (-1488));
      JsDocInfoParser jsDocInfoParser0 = new JsDocInfoParser(jsDocTokenStream0, (Comment) null, node0, config0, toolErrorReporter0);
      JsDocToken jsDocToken0 = JsDocToken.LB;
      jsDocInfoParser0.parseAndRecordTypeNode(jsDocToken0);
      assertEquals(0, node0.getSourcePosition());
      assertEquals(1, node0.getLength());
  }

  @Test(timeout = 4000)
  public void test48()  throws Throwable  {
      JsDocTokenStream jsDocTokenStream0 = new JsDocTokenStream("\"y]1ag+3");
      Node node0 = Node.newString("\"y]1ag+3");
      JSDocInfo jSDocInfo0 = new JSDocInfo();
      Set<String> set0 = jSDocInfo0.getModifies();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT3;
      Config config0 = new Config(set0, set0, false, config_LanguageMode0, false);
      ToolErrorReporter toolErrorReporter0 = new ToolErrorReporter(false);
      JsDocInfoParser jsDocInfoParser0 = new JsDocInfoParser(jsDocTokenStream0, (Comment) null, node0, config0, toolErrorReporter0);
      JsDocToken jsDocToken0 = JsDocToken.LB;
      Node node1 = jsDocInfoParser0.parseAndRecordTypeNode(jsDocToken0);
      assertEquals(308, node1.getType());
      assertNotNull(node1);
      assertTrue(node1.hasOneChild());
  }

  @Test(timeout = 4000)
  public void test49()  throws Throwable  {
      Node node0 = JsDocInfoParser.parseTypeString("{}G69?");
      assertNull(node0);
  }

  @Test(timeout = 4000)
  public void test50()  throws Throwable  {
      Node node0 = JsDocInfoParser.parseTypeString("{&SG8zcF$ir4:&");
      assertNull(node0);
  }

  @Test(timeout = 4000)
  public void test51()  throws Throwable  {
      JsDocTokenStream jsDocTokenStream0 = new JsDocTokenStream("*J&OZ", 9, 150);
      Token.CommentType token_CommentType0 = Token.CommentType.BLOCK_COMMENT;
      Comment comment0 = new Comment((-1), 160, token_CommentType0, "language version");
      Node node0 = Node.newString(17, "language version");
      LinkedHashSet<String> linkedHashSet0 = new LinkedHashSet<String>();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT3;
      Config config0 = new Config(linkedHashSet0, linkedHashSet0, true, config_LanguageMode0, true);
      ToolErrorReporter toolErrorReporter0 = new ToolErrorReporter(false);
      JsDocInfoParser jsDocInfoParser0 = new JsDocInfoParser(jsDocTokenStream0, comment0, node0, config0, toolErrorReporter0);
      JsDocToken jsDocToken0 = JsDocToken.EOL;
      jsDocInfoParser0.parseAndRecordTypeNode(jsDocToken0);
      boolean boolean0 = jsDocInfoParser0.parse();
      assertFalse(boolean0);
  }
}
