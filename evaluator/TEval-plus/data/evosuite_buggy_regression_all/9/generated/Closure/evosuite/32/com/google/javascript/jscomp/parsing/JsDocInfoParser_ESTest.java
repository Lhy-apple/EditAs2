/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 04:46:20 GMT 2023
 */

package com.google.javascript.jscomp.parsing;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.google.javascript.jscomp.parsing.Config;
import com.google.javascript.jscomp.parsing.JsDocInfoParser;
import com.google.javascript.jscomp.parsing.JsDocTokenStream;
import com.google.javascript.rhino.JSDocInfo;
import com.google.javascript.rhino.Node;
import com.google.javascript.rhino.head.Context;
import com.google.javascript.rhino.head.ErrorReporter;
import com.google.javascript.rhino.head.Token;
import com.google.javascript.rhino.head.ast.Comment;
import com.google.javascript.rhino.head.ast.ErrorCollector;
import com.google.javascript.rhino.head.tools.ToolErrorReporter;
import java.nio.charset.Charset;
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
      TreeSet<String> treeSet0 = new TreeSet<String>();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5_STRICT;
      Config config0 = new Config(treeSet0, treeSet0, true, config_LanguageMode0, true);
      ErrorCollector errorCollector0 = new ErrorCollector();
      Node node0 = Node.newString("language version");
      JsDocTokenStream jsDocTokenStream0 = new JsDocTokenStream("language version", 44);
      JsDocInfoParser jsDocInfoParser0 = new JsDocInfoParser(jsDocTokenStream0, (Comment) null, node0, config0, errorCollector0);
      boolean boolean0 = jsDocInfoParser0.hasParsedJSDocInfo();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      TreeSet<String> treeSet0 = new TreeSet<String>();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5_STRICT;
      Config config0 = new Config(treeSet0, treeSet0, true, config_LanguageMode0, true);
      ErrorCollector errorCollector0 = new ErrorCollector();
      JsDocTokenStream jsDocTokenStream0 = new JsDocTokenStream("=o$e$Kt");
      Node node0 = JsDocInfoParser.parseTypeString("error reporter");
      assertNotNull(node0);
      
      JsDocInfoParser jsDocInfoParser0 = new JsDocInfoParser(jsDocTokenStream0, (Comment) null, node0, config0, errorCollector0);
      jsDocInfoParser0.getFileOverviewJSDocInfo();
      assertEquals(0, node0.getSourcePosition());
      assertTrue(node0.isString());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      TreeSet<String> treeSet0 = new TreeSet<String>();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5_STRICT;
      Config config0 = new Config(treeSet0, treeSet0, true, config_LanguageMode0, true);
      ErrorCollector errorCollector0 = new ErrorCollector();
      JsDocTokenStream jsDocTokenStream0 = new JsDocTokenStream("*]He^}j)kz aBCd");
      Node node0 = JsDocInfoParser.parseTypeString("*]He^}j)kz aBCd");
      assertNotNull(node0);
      
      JsDocInfoParser jsDocInfoParser0 = new JsDocInfoParser(jsDocTokenStream0, (Comment) null, node0, config0, errorCollector0);
      Node.FileLevelJsDocBuilder node_FileLevelJsDocBuilder0 = node0.getJsDocBuilderForNode();
      jsDocInfoParser0.setFileLevelJsDocBuilder(node_FileLevelJsDocBuilder0);
      assertEquals(302, node0.getType());
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      TreeSet<String> treeSet0 = new TreeSet<String>();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5_STRICT;
      Config config0 = new Config(treeSet0, treeSet0, true, config_LanguageMode0, true);
      ErrorCollector errorCollector0 = new ErrorCollector();
      JsDocTokenStream jsDocTokenStream0 = new JsDocTokenStream("=o$e$Kt");
      Node node0 = JsDocInfoParser.parseTypeString("error reporter");
      assertNotNull(node0);
      
      JsDocInfoParser jsDocInfoParser0 = new JsDocInfoParser(jsDocTokenStream0, (Comment) null, node0, config0, errorCollector0);
      jsDocInfoParser0.setFileOverviewJSDocInfo((JSDocInfo) null);
      assertEquals(0, node0.getSourcePosition());
      assertEquals(0, node0.getLineno());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      TreeSet<String> treeSet0 = new TreeSet<String>();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5_STRICT;
      Config config0 = new Config(treeSet0, treeSet0, false, config_LanguageMode0, false);
      ErrorCollector errorCollector0 = new ErrorCollector();
      JsDocTokenStream jsDocTokenStream0 = new JsDocTokenStream("\n");
      Node node0 = JsDocInfoParser.parseTypeString("language version");
      jsDocTokenStream0.ungetChar(50);
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
  public void test05()  throws Throwable  {
      Node node0 = JsDocInfoParser.parseTypeString("(dYP|gX0]iT8En4-z^!m");
      assertNull(node0);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Node node0 = JsDocInfoParser.parseTypeString("?{piT3+8W");
      assertNull(node0);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      JsDocTokenStream jsDocTokenStream0 = new JsDocTokenStream("(dYP|gX0]iT8En4-z^!m", 0, 44);
      TreeSet<String> treeSet0 = new TreeSet<String>();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5_STRICT;
      Config config0 = new Config(treeSet0, treeSet0, true, config_LanguageMode0, true);
      JsDocInfoParser jsDocInfoParser0 = new JsDocInfoParser(jsDocTokenStream0, (Comment) null, (Node) null, config0, (ErrorReporter) null);
      JSDocInfo jSDocInfo0 = jsDocInfoParser0.retrieveAndResetParsedJSDocInfo();
      assertNull(jSDocInfo0);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      TreeSet<String> treeSet0 = new TreeSet<String>();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5;
      Config config0 = new Config(treeSet0, treeSet0, true, config_LanguageMode0, true);
      JsDocTokenStream jsDocTokenStream0 = new JsDocTokenStream("@");
      ToolErrorReporter toolErrorReporter0 = new ToolErrorReporter(true);
      JsDocInfoParser jsDocInfoParser0 = new JsDocInfoParser(jsDocTokenStream0, (Comment) null, (Node) null, config0, toolErrorReporter0);
      boolean boolean0 = jsDocInfoParser0.parse();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Node node0 = JsDocInfoParser.parseTypeString("5;|0");
      assertNotNull(node0);
      
      Token.CommentType token_CommentType0 = Token.CommentType.JSDOC;
      Comment comment0 = new Comment(2, 50, token_CommentType0, "5;|0");
      LinkedHashSet<String> linkedHashSet0 = new LinkedHashSet<String>();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT3;
      JsDocTokenStream jsDocTokenStream0 = new JsDocTokenStream(">){[<AJn", 2, 16);
      Config config0 = new Config(linkedHashSet0, linkedHashSet0, true, config_LanguageMode0, true);
      Context context0 = new Context();
      ErrorReporter errorReporter0 = context0.getErrorReporter();
      JsDocInfoParser jsDocInfoParser0 = new JsDocInfoParser(jsDocTokenStream0, comment0, node0, config0, errorReporter0);
      jsDocInfoParser0.parse();
      assertEquals(301, node0.getType());
      assertFalse(node0.hasOneChild());
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      TreeSet<String> treeSet0 = new TreeSet<String>();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5_STRICT;
      Config config0 = new Config(treeSet0, treeSet0, true, config_LanguageMode0, true);
      ErrorCollector errorCollector0 = new ErrorCollector();
      Node node0 = Node.newString(9, "language version", 3, 11);
      JsDocTokenStream jsDocTokenStream0 = new JsDocTokenStream(")", 52);
      node0.setSourceFileForTesting("error reporter");
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
  public void test11()  throws Throwable  {
      TreeSet<String> treeSet0 = new TreeSet<String>();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5;
      Config config0 = new Config(treeSet0, treeSet0, false, config_LanguageMode0, false);
      ErrorCollector errorCollector0 = new ErrorCollector();
      JsDocTokenStream jsDocTokenStream0 = new JsDocTokenStream("@");
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
  public void test12()  throws Throwable  {
      TreeSet<String> treeSet0 = new TreeSet<String>();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5_STRICT;
      Config config0 = new Config(treeSet0, treeSet0, false, config_LanguageMode0, false);
      ErrorCollector errorCollector0 = new ErrorCollector();
      JsDocTokenStream jsDocTokenStream0 = new JsDocTokenStream("*]He^}j)kz aBCd");
      Node node0 = JsDocInfoParser.parseTypeString("*]He^}j)kz aBCd");
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
  public void test13()  throws Throwable  {
      JsDocTokenStream jsDocTokenStream0 = new JsDocTokenStream("!#xC|h(H{;");
      TreeSet<String> treeSet0 = new TreeSet<String>();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5_STRICT;
      Config config0 = new Config(treeSet0, treeSet0, true, config_LanguageMode0, true);
      JsDocInfoParser jsDocInfoParser0 = new JsDocInfoParser(jsDocTokenStream0, (Comment) null, (Node) null, config0, (ErrorReporter) null);
      // Undeclared exception!
      try { 
        jsDocInfoParser0.parse();
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.parsing.JsDocInfoParser$ErrorReporterParser", e);
      }
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      Charset charset0 = Charset.defaultCharset();
      Set<String> set0 = charset0.aliases();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5_STRICT;
      Config config0 = new Config(set0, set0, true, config_LanguageMode0, true);
      ErrorCollector errorCollector0 = new ErrorCollector();
      Node node0 = JsDocInfoParser.parseTypeString("language version");
      JsDocTokenStream jsDocTokenStream0 = new JsDocTokenStream(",!{8MFnfX<{", 8);
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
  public void test15()  throws Throwable  {
      TreeSet<String> treeSet0 = new TreeSet<String>();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5_STRICT;
      Config config0 = new Config(treeSet0, treeSet0, true, config_LanguageMode0, true);
      ErrorCollector errorCollector0 = new ErrorCollector();
      JsDocTokenStream jsDocTokenStream0 = new JsDocTokenStream(":f3");
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
  public void test16()  throws Throwable  {
      JsDocTokenStream jsDocTokenStream0 = new JsDocTokenStream("[S@f-CS", 35);
      LinkedHashSet<String> linkedHashSet0 = new LinkedHashSet<String>();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT3;
      Config config0 = new Config(linkedHashSet0, linkedHashSet0, true, config_LanguageMode0, true);
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
  public void test17()  throws Throwable  {
      TreeSet<String> treeSet0 = new TreeSet<String>();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5;
      Config config0 = new Config(treeSet0, treeSet0, true, config_LanguageMode0, true);
      JsDocTokenStream jsDocTokenStream0 = new JsDocTokenStream("{Frr\"Vw:sQ");
      JsDocInfoParser jsDocInfoParser0 = new JsDocInfoParser(jsDocTokenStream0, (Comment) null, (Node) null, config0, (ErrorReporter) null);
      // Undeclared exception!
      try { 
        jsDocInfoParser0.parse();
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.parsing.JsDocInfoParser$ErrorReporterParser", e);
      }
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      JsDocTokenStream jsDocTokenStream0 = new JsDocTokenStream("(dYP|gX0]iT8En4-z^!m", 0, 44);
      TreeSet<String> treeSet0 = new TreeSet<String>();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5_STRICT;
      Config config0 = new Config(treeSet0, treeSet0, true, config_LanguageMode0, true);
      JsDocInfoParser jsDocInfoParser0 = new JsDocInfoParser(jsDocTokenStream0, (Comment) null, (Node) null, config0, (ErrorReporter) null);
      // Undeclared exception!
      try { 
        jsDocInfoParser0.parse();
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.parsing.JsDocInfoParser$ErrorReporterParser", e);
      }
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      TreeSet<String> treeSet0 = new TreeSet<String>();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5_STRICT;
      Config config0 = new Config(treeSet0, treeSet0, true, config_LanguageMode0, true);
      ToolErrorReporter toolErrorReporter0 = new ToolErrorReporter(true);
      JsDocTokenStream jsDocTokenStream0 = new JsDocTokenStream(".<", 12, 58);
      Token.CommentType token_CommentType0 = Token.CommentType.JSDOC;
      Comment comment0 = new Comment((-1718), 5164, token_CommentType0, "error reporter");
      Node node0 = Node.newNumber((-907.61750244736));
      JsDocInfoParser jsDocInfoParser0 = new JsDocInfoParser(jsDocTokenStream0, comment0, node0, config0, toolErrorReporter0);
      boolean boolean0 = jsDocInfoParser0.parse();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      JsDocTokenStream jsDocTokenStream0 = new JsDocTokenStream("?{piT3+8W", 1358);
      LinkedHashSet<String> linkedHashSet0 = new LinkedHashSet<String>();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT3;
      Config config0 = new Config(linkedHashSet0, linkedHashSet0, true, config_LanguageMode0, true);
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
  public void test21()  throws Throwable  {
      TreeSet<String> treeSet0 = new TreeSet<String>();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5;
      Config config0 = new Config(treeSet0, treeSet0, true, config_LanguageMode0, true);
      ErrorCollector errorCollector0 = new ErrorCollector();
      Node node0 = JsDocInfoParser.parseTypeString("*]He^}j)kz aBCd");
      JsDocTokenStream jsDocTokenStream0 = new JsDocTokenStream("|=", 864, 31);
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
  public void test22()  throws Throwable  {
      TreeSet<String> treeSet0 = new TreeSet<String>();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5_STRICT;
      Config config0 = new Config(treeSet0, treeSet0, true, config_LanguageMode0, true);
      ErrorCollector errorCollector0 = new ErrorCollector();
      JsDocTokenStream jsDocTokenStream0 = new JsDocTokenStream("*]He^}j)kz aBCd");
      Node node0 = JsDocInfoParser.parseTypeString("*]He^}j)kz aBCd");
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
  public void test23()  throws Throwable  {
      TreeSet<String> treeSet0 = new TreeSet<String>();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5_STRICT;
      Config config0 = new Config(treeSet0, treeSet0, true, config_LanguageMode0, true);
      ErrorCollector errorCollector0 = new ErrorCollector();
      JsDocTokenStream jsDocTokenStream0 = new JsDocTokenStream("}d<)(hEgHr<x?");
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
  public void test24()  throws Throwable  {
      TreeSet<String> treeSet0 = new TreeSet<String>();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5;
      Config config0 = new Config(treeSet0, treeSet0, true, config_LanguageMode0, true);
      ErrorCollector errorCollector0 = new ErrorCollector();
      JsDocTokenStream jsDocTokenStream0 = new JsDocTokenStream("=o$e$t");
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
  public void test25()  throws Throwable  {
      TreeSet<String> treeSet0 = new TreeSet<String>();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5_STRICT;
      Config config0 = new Config(treeSet0, treeSet0, true, config_LanguageMode0, true);
      ErrorCollector errorCollector0 = new ErrorCollector();
      JsDocTokenStream jsDocTokenStream0 = new JsDocTokenStream("Node tree inequality:\nTree1:\n");
      Node node0 = JsDocInfoParser.parseTypeString("M<O%_M");
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
  public void test26()  throws Throwable  {
      TreeSet<String> treeSet0 = new TreeSet<String>();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT3;
      Config config0 = new Config(treeSet0, treeSet0, true, config_LanguageMode0, true);
      ErrorCollector errorCollector0 = new ErrorCollector();
      JsDocTokenStream jsDocTokenStream0 = new JsDocTokenStream("Could not dicver accessible methods ofXclass ");
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
  public void test27()  throws Throwable  {
      Node node0 = JsDocInfoParser.parseTypeString("?]Z{i1");
      assertNotNull(node0);
      assertEquals(304, node0.getType());
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      Node node0 = JsDocInfoParser.parseTypeString("!#xC|h(H{;");
      assertEquals(301, node0.getType());
      assertNotNull(node0);
      assertFalse(node0.hasOneChild());
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      Node node0 = JsDocInfoParser.parseTypeString("(c,~6q?~@n?;");
      assertNull(node0);
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      Node node0 = JsDocInfoParser.parseTypeString("j!l`7,7q]nXMGA~6\"{6");
      assertNotNull(node0);
      assertEquals(306, node0.getType());
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      Node node0 = JsDocInfoParser.parseTypeString("function");
      assertNull(node0);
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      Node node0 = JsDocInfoParser.parseTypeString("null");
      assertNotNull(node0);
      assertEquals(0, node0.getSourcePosition());
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      Node node0 = JsDocInfoParser.parseTypeString("undefined");
      assertNotNull(node0);
      assertEquals(0, node0.getSourcePosition());
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      Node node0 = JsDocInfoParser.parseTypeString("l|");
      assertNull(node0);
  }

  @Test(timeout = 4000)
  public void test35()  throws Throwable  {
      Node node0 = JsDocInfoParser.parseTypeString("(Ljava/lang/String;)Ljava/lang/StringBuffer;");
      assertNotNull(node0);
      assertEquals(301, node0.getType());
      assertEquals(1, node0.getChildCount());
  }

  @Test(timeout = 4000)
  public void test36()  throws Throwable  {
      Node node0 = JsDocInfoParser.parseTypeString("[1B4r,B@921<~");
      assertNull(node0);
  }

  @Test(timeout = 4000)
  public void test37()  throws Throwable  {
      Node node0 = JsDocInfoParser.parseTypeString("[");
      assertNull(node0);
  }

  @Test(timeout = 4000)
  public void test38()  throws Throwable  {
      Node node0 = JsDocInfoParser.parseTypeString("[ ~2]");
      assertEquals(308, node0.getType());
      assertNotNull(node0);
      assertTrue(node0.hasChildren());
  }

  @Test(timeout = 4000)
  public void test39()  throws Throwable  {
      Node node0 = JsDocInfoParser.parseTypeString("{");
      assertNull(node0);
  }

  @Test(timeout = 4000)
  public void test40()  throws Throwable  {
      Node node0 = JsDocInfoParser.parseTypeString("{ei}@?_");
      assertEquals(309, node0.getType());
      assertNotNull(node0);
      assertTrue(node0.hasChildren());
  }

  @Test(timeout = 4000)
  public void test41()  throws Throwable  {
      Node node0 = JsDocInfoParser.parseTypeString("{Fr\"Vuw,:sQ");
      assertNull(node0);
  }

  @Test(timeout = 4000)
  public void test42()  throws Throwable  {
      TreeSet<String> treeSet0 = new TreeSet<String>();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT3;
      Config config0 = new Config(treeSet0, treeSet0, false, config_LanguageMode0, false);
      ErrorCollector errorCollector0 = new ErrorCollector();
      JsDocTokenStream jsDocTokenStream0 = new JsDocTokenStream("\n");
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
}