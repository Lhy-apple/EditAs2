/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 21:10:51 GMT 2023
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
import com.google.javascript.rhino.jstype.SimpleSourceFile;
import java.nio.charset.Charset;
import java.util.LinkedHashSet;
import java.util.Locale;
import java.util.Set;
import java.util.TreeSet;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class JsDocInfoParser_ESTest extends JsDocInfoParser_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      ErrorCollector errorCollector0 = new ErrorCollector();
      JsDocTokenStream jsDocTokenStream0 = new JsDocTokenStream("_q{#;~*/&3zF&s+'?r;");
      TreeSet<String> treeSet0 = new TreeSet<String>();
      Node node0 = Node.newString("_q{#;~*/&3zF&s+'?r;");
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5_STRICT;
      Config config0 = new Config(treeSet0, treeSet0, true, config_LanguageMode0, true);
      JsDocInfoParser jsDocInfoParser0 = new JsDocInfoParser(jsDocTokenStream0, (Comment) null, node0, config0, errorCollector0);
      boolean boolean0 = jsDocInfoParser0.parse();
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      JsDocTokenStream jsDocTokenStream0 = new JsDocTokenStream("*}hu#*Z{\"[_iZL");
      TreeSet<String> treeSet0 = new TreeSet<String>();
      Node node0 = Node.newString("*}hu#*Z{\"[_iZL");
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5;
      Config config0 = new Config(treeSet0, treeSet0, true, config_LanguageMode0, true);
      ErrorCollector errorCollector0 = new ErrorCollector();
      JsDocInfoParser jsDocInfoParser0 = new JsDocInfoParser(jsDocTokenStream0, (Comment) null, node0, config0, errorCollector0);
      boolean boolean0 = jsDocInfoParser0.hasParsedJSDocInfo();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      LinkedHashSet<String> linkedHashSet0 = new LinkedHashSet<String>();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT3;
      Config config0 = new Config(linkedHashSet0, linkedHashSet0, true, config_LanguageMode0, true);
      ErrorCollector errorCollector0 = new ErrorCollector();
      JsDocInfoParser jsDocInfoParser0 = new JsDocInfoParser((JsDocTokenStream) null, (Comment) null, (Node) null, config0, errorCollector0);
      JSDocInfo jSDocInfo0 = jsDocInfoParser0.getFileOverviewJSDocInfo();
      assertNull(jSDocInfo0);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      TreeSet<String> treeSet0 = new TreeSet<String>();
      Node node0 = Node.newString("?");
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5;
      Config config0 = new Config(treeSet0, treeSet0, true, config_LanguageMode0, true);
      ErrorCollector errorCollector0 = new ErrorCollector();
      JsDocTokenStream jsDocTokenStream0 = new JsDocTokenStream("?", 1);
      JsDocInfoParser jsDocInfoParser0 = new JsDocInfoParser(jsDocTokenStream0, (Comment) null, node0, config0, errorCollector0);
      Node.FileLevelJsDocBuilder node_FileLevelJsDocBuilder0 = node0.new FileLevelJsDocBuilder();
      jsDocInfoParser0.setFileLevelJsDocBuilder(node_FileLevelJsDocBuilder0);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      TreeSet<String> treeSet0 = new TreeSet<String>();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5;
      Config config0 = new Config(treeSet0, treeSet0, true, config_LanguageMode0, true);
      ErrorCollector errorCollector0 = new ErrorCollector();
      JsDocTokenStream jsDocTokenStream0 = new JsDocTokenStream("...");
      JsDocInfoParser jsDocInfoParser0 = new JsDocInfoParser(jsDocTokenStream0, (Comment) null, (Node) null, config0, errorCollector0);
      jsDocInfoParser0.setFileOverviewJSDocInfo((JSDocInfo) null);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      Node node0 = JsDocInfoParser.parseTypeString("isNaN");
      assertNotNull(node0);
      
      Locale locale0 = Locale.ITALY;
      Set<String> set0 = locale0.getUnicodeLocaleAttributes();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5;
      Config config0 = new Config(set0, set0, true, config_LanguageMode0, true);
      Context context0 = Context.enter();
      ErrorReporter errorReporter0 = context0.getErrorReporter();
      JsDocTokenStream jsDocTokenStream0 = new JsDocTokenStream("@*kS", (-498));
      Token.CommentType token_CommentType0 = Token.CommentType.HTML;
      Comment comment0 = new Comment(1, 140, token_CommentType0, "45r");
      JsDocInfoParser jsDocInfoParser0 = new JsDocInfoParser(jsDocTokenStream0, comment0, node0, config0, errorReporter0);
      jsDocInfoParser0.parse();
      assertEquals(0, node0.getSourcePosition());
      assertEquals(40, node0.getType());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Node node0 = JsDocInfoParser.parseTypeString("(Ljava/lang/Object;)J");
      assertTrue(node0.hasChildren());
      assertNotNull(node0);
      assertEquals(301, node0.getType());
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Node node0 = JsDocInfoParser.parseTypeString("{lQ:TH");
      assertNull(node0);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      TreeSet<String> treeSet0 = new TreeSet<String>();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5;
      Config config0 = new Config(treeSet0, treeSet0, true, config_LanguageMode0, true);
      ErrorCollector errorCollector0 = new ErrorCollector();
      JsDocTokenStream jsDocTokenStream0 = new JsDocTokenStream("...");
      JsDocInfoParser jsDocInfoParser0 = new JsDocInfoParser(jsDocTokenStream0, (Comment) null, (Node) null, config0, errorCollector0);
      JSDocInfo jSDocInfo0 = jsDocInfoParser0.retrieveAndResetParsedJSDocInfo();
      assertNull(jSDocInfo0);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      JsDocTokenStream jsDocTokenStream0 = new JsDocTokenStream("...", (-3658), 463);
      Node node0 = new Node((-372));
      SimpleSourceFile simpleSourceFile0 = new SimpleSourceFile("...", true);
      node0.setStaticSourceFile(simpleSourceFile0);
      TreeSet<String> treeSet0 = new TreeSet<String>();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT3;
      Config config0 = new Config(treeSet0, treeSet0, true, config_LanguageMode0, true);
      Context context0 = new Context();
      ErrorReporter errorReporter0 = context0.getErrorReporter();
      JsDocInfoParser jsDocInfoParser0 = new JsDocInfoParser(jsDocTokenStream0, (Comment) null, node0, config0, errorReporter0);
      boolean boolean0 = jsDocInfoParser0.parse();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      TreeSet<String> treeSet0 = new TreeSet<String>();
      ErrorCollector errorCollector0 = new ErrorCollector();
      JsDocTokenStream jsDocTokenStream0 = new JsDocTokenStream("@: cM7ur7");
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5_STRICT;
      Config config0 = new Config(treeSet0, treeSet0, false, config_LanguageMode0, false);
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
  public void test11()  throws Throwable  {
      ErrorCollector errorCollector0 = new ErrorCollector();
      JsDocTokenStream jsDocTokenStream0 = new JsDocTokenStream("_q{#;~*/&3zF&s+'?r;");
      TreeSet<String> treeSet0 = new TreeSet<String>();
      Node node0 = Node.newString("_q{#;~*/&3zF&s+'?r;");
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT3;
      Config config0 = new Config(treeSet0, treeSet0, false, config_LanguageMode0, true);
      JsDocInfoParser jsDocInfoParser0 = new JsDocInfoParser(jsDocTokenStream0, (Comment) null, node0, config0, errorCollector0);
      boolean boolean0 = jsDocInfoParser0.parse();
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      TreeSet<String> treeSet0 = new TreeSet<String>();
      Node node0 = Node.newString("fW\n");
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5_STRICT;
      Config config0 = new Config(treeSet0, treeSet0, false, config_LanguageMode0, false);
      ErrorCollector errorCollector0 = new ErrorCollector();
      JsDocTokenStream jsDocTokenStream0 = new JsDocTokenStream("fW\n");
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
      JsDocTokenStream jsDocTokenStream0 = new JsDocTokenStream("*}hu#*Z{\"[_iZL");
      TreeSet<String> treeSet0 = new TreeSet<String>();
      Node node0 = Node.newString("*}hu#*Z{\"[_iZL");
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5;
      Config config0 = new Config(treeSet0, treeSet0, false, config_LanguageMode0, false);
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
  public void test14()  throws Throwable  {
      Node node0 = Node.newString("[%;oVBX)eL");
      LinkedHashSet<String> linkedHashSet0 = new LinkedHashSet<String>();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5_STRICT;
      Config config0 = new Config(linkedHashSet0, linkedHashSet0, true, config_LanguageMode0, true);
      ErrorCollector errorCollector0 = new ErrorCollector();
      JsDocTokenStream jsDocTokenStream0 = new JsDocTokenStream("!Kr%CF4{d", 49);
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
      Node node0 = Node.newString("[%;oVBX)eL");
      LinkedHashSet<String> linkedHashSet0 = new LinkedHashSet<String>();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5_STRICT;
      Config config0 = new Config(linkedHashSet0, linkedHashSet0, true, config_LanguageMode0, true);
      ErrorCollector errorCollector0 = new ErrorCollector();
      JsDocTokenStream jsDocTokenStream0 = new JsDocTokenStream(",+rDjx9<ND[L%M");
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
  public void test16()  throws Throwable  {
      TreeSet<String> treeSet0 = new TreeSet<String>();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5;
      Config config0 = new Config(treeSet0, treeSet0, true, config_LanguageMode0, true);
      ErrorCollector errorCollector0 = new ErrorCollector();
      JsDocTokenStream jsDocTokenStream0 = new JsDocTokenStream(":");
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
      JsDocTokenStream jsDocTokenStream0 = new JsDocTokenStream(">y5HP%gg}YM", 37, 37);
      Node node0 = Node.newString(">y5HP%gg}YM");
      Charset charset0 = Charset.defaultCharset();
      Set<String> set0 = charset0.aliases();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5;
      Config config0 = new Config(set0, set0, true, config_LanguageMode0, true);
      Context context0 = new Context();
      ErrorReporter errorReporter0 = context0.getErrorReporter();
      JsDocInfoParser jsDocInfoParser0 = new JsDocInfoParser(jsDocTokenStream0, (Comment) null, node0, config0, errorReporter0);
      boolean boolean0 = jsDocInfoParser0.parse();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      JsDocTokenStream jsDocTokenStream0 = new JsDocTokenStream("[-$\n");
      TreeSet<String> treeSet0 = new TreeSet<String>();
      Node node0 = Node.newString("[-$\n");
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5_STRICT;
      Config config0 = new Config(treeSet0, treeSet0, true, config_LanguageMode0, true);
      JsDocInfoParser jsDocInfoParser0 = new JsDocInfoParser(jsDocTokenStream0, (Comment) null, node0, config0, (ErrorReporter) null);
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
      Node node0 = Node.newString("{r:");
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT3;
      Config config0 = new Config(treeSet0, treeSet0, true, config_LanguageMode0, true);
      ErrorCollector errorCollector0 = new ErrorCollector();
      JsDocTokenStream jsDocTokenStream0 = new JsDocTokenStream("{r:");
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
  public void test20()  throws Throwable  {
      TreeSet<String> treeSet0 = new TreeSet<String>();
      Node node0 = Node.newString("([Ljava/lang/Object;Ljava/lang/String;Lorg/mozilla/javascript/Context;Lorg/mozilla/javascript/Scriptable;)Ljava/lang/Object;");
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5_STRICT;
      Config config0 = new Config(treeSet0, treeSet0, true, config_LanguageMode0, true);
      ErrorCollector errorCollector0 = new ErrorCollector();
      JsDocTokenStream jsDocTokenStream0 = new JsDocTokenStream("([Ljava/lang/Object;Ljava/lang/String;Lorg/mozilla/javascript/Context;Lorg/mozilla/javascript/Scriptable;)Ljava/lang/Object;");
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
  public void test21()  throws Throwable  {
      TreeSet<String> treeSet0 = new TreeSet<String>();
      Node node0 = Node.newString("?");
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5;
      Config config0 = new Config(treeSet0, treeSet0, true, config_LanguageMode0, true);
      ErrorCollector errorCollector0 = new ErrorCollector();
      JsDocTokenStream jsDocTokenStream0 = new JsDocTokenStream("?", 1);
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
      JsDocTokenStream jsDocTokenStream0 = new JsDocTokenStream("|}hu#*Z{\"[_i5L");
      TreeSet<String> treeSet0 = new TreeSet<String>();
      Node node0 = JsDocInfoParser.parseTypeString("mN");
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT3;
      Config config0 = new Config(treeSet0, treeSet0, true, config_LanguageMode0, false);
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
  public void test23()  throws Throwable  {
      TreeSet<String> treeSet0 = new TreeSet<String>();
      Node node0 = Node.newString("]8j\"");
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT3;
      Config config0 = new Config(treeSet0, treeSet0, true, config_LanguageMode0, true);
      ErrorCollector errorCollector0 = new ErrorCollector();
      JsDocTokenStream jsDocTokenStream0 = new JsDocTokenStream("]8j\"");
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
  public void test24()  throws Throwable  {
      JsDocTokenStream jsDocTokenStream0 = new JsDocTokenStream("*}hu#*Z{\"[_iZL");
      TreeSet<String> treeSet0 = new TreeSet<String>();
      Node node0 = Node.newString("*}hu#*Z{\"[_iZL");
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT3;
      Config config0 = new Config(treeSet0, treeSet0, true, config_LanguageMode0, true);
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
  public void test25()  throws Throwable  {
      TreeSet<String> treeSet0 = new TreeSet<String>();
      Node node0 = Node.newString("?");
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5;
      Config config0 = new Config(treeSet0, treeSet0, true, config_LanguageMode0, true);
      JsDocTokenStream jsDocTokenStream0 = new JsDocTokenStream(")WL*d/J");
      JsDocInfoParser jsDocInfoParser0 = new JsDocInfoParser(jsDocTokenStream0, (Comment) null, node0, config0, (ErrorReporter) null);
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
  public void test26()  throws Throwable  {
      JsDocTokenStream jsDocTokenStream0 = new JsDocTokenStream("=");
      Token.CommentType token_CommentType0 = Token.CommentType.JSDOC;
      Comment comment0 = new Comment(31, 38, token_CommentType0, "=");
      Charset charset0 = Charset.defaultCharset();
      Set<String> set0 = charset0.aliases();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT3;
      Config config0 = new Config(set0, set0, true, config_LanguageMode0, true);
      ToolErrorReporter toolErrorReporter0 = new ToolErrorReporter(true);
      JsDocInfoParser jsDocInfoParser0 = new JsDocInfoParser(jsDocTokenStream0, comment0, (Node) null, config0, toolErrorReporter0);
      boolean boolean0 = jsDocInfoParser0.parse();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      TreeSet<String> treeSet0 = new TreeSet<String>();
      Config.LanguageMode config_LanguageMode0 = Config.LanguageMode.ECMASCRIPT5;
      Config config0 = new Config(treeSet0, treeSet0, true, config_LanguageMode0, false);
      ErrorCollector errorCollector0 = new ErrorCollector();
      JsDocTokenStream jsDocTokenStream0 = new JsDocTokenStream("BadGtype annotation. ");
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
  public void test28()  throws Throwable  {
      Node node0 = JsDocInfoParser.parseTypeString("9Fe89.|GSHr");
      assertFalse(node0.hasOneChild());
      assertEquals(301, node0.getType());
      assertNotNull(node0);
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      Node node0 = JsDocInfoParser.parseTypeString("?");
      assertNull(node0);
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      Node node0 = JsDocInfoParser.parseTypeString("!?");
      assertNull(node0);
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      Node node0 = JsDocInfoParser.parseTypeString("P4?;8#v.9YOJ`gM:#R");
      assertEquals(304, node0.getType());
      assertNotNull(node0);
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      Node node0 = JsDocInfoParser.parseTypeString("'M!U<>xZKkC0ZZlt");
      assertNotNull(node0);
      assertEquals(306, node0.getType());
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      Node node0 = JsDocInfoParser.parseTypeString("*}hu#*Z{\"[_iZL");
      assertEquals(302, node0.getType());
      assertNotNull(node0);
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      Node node0 = JsDocInfoParser.parseTypeString("function");
      assertNull(node0);
  }

  @Test(timeout = 4000)
  public void test35()  throws Throwable  {
      Node node0 = JsDocInfoParser.parseTypeString("null");
      assertEquals(0, node0.getSourcePosition());
      assertEquals(4, node0.getLength());
      assertNotNull(node0);
  }

  @Test(timeout = 4000)
  public void test36()  throws Throwable  {
      Node node0 = JsDocInfoParser.parseTypeString("undefined");
      assertEquals(40, node0.getType());
      assertNotNull(node0);
      assertEquals(0, node0.getSourcePosition());
  }

  @Test(timeout = 4000)
  public void test37()  throws Throwable  {
      Node node0 = JsDocInfoParser.parseTypeString("[-$\n");
      assertNull(node0);
  }

  @Test(timeout = 4000)
  public void test38()  throws Throwable  {
      Node node0 = JsDocInfoParser.parseTypeString("(V5x|H75tvZ");
      assertNull(node0);
  }

  @Test(timeout = 4000)
  public void test39()  throws Throwable  {
      Node node0 = JsDocInfoParser.parseTypeString("(");
      assertNull(node0);
  }

  @Test(timeout = 4000)
  public void test40()  throws Throwable  {
      Node node0 = JsDocInfoParser.parseTypeString("[)\n");
      assertNull(node0);
  }

  @Test(timeout = 4000)
  public void test41()  throws Throwable  {
      Node node0 = JsDocInfoParser.parseTypeString("{AOt1}Y");
      assertNotNull(node0);
      assertEquals(309, node0.getType());
      assertTrue(node0.hasChildren());
  }

  @Test(timeout = 4000)
  public void test42()  throws Throwable  {
      Node node0 = JsDocInfoParser.parseTypeString("{,`TP&&(");
      assertNull(node0);
  }

  @Test(timeout = 4000)
  public void test43()  throws Throwable  {
      Node node0 = JsDocInfoParser.parseTypeString("{proxy:");
      assertNull(node0);
  }
}