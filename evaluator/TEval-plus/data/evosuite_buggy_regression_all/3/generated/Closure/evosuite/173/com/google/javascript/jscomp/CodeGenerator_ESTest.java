/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 17:24:03 GMT 2023
 */

package com.google.javascript.jscomp;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.google.javascript.jscomp.CodeConsumer;
import com.google.javascript.jscomp.CodeGenerator;
import com.google.javascript.jscomp.Compiler;
import com.google.javascript.jscomp.CompilerOptions;
import com.google.javascript.jscomp.LightweightMessageFormatter;
import com.google.javascript.jscomp.LoggerErrorManager;
import com.google.javascript.jscomp.Normalize;
import com.google.javascript.rhino.Node;
import java.nio.charset.Charset;
import java.nio.charset.CharsetEncoder;
import java.util.logging.Logger;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.mock.java.io.MockPrintStream;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class CodeGenerator_ESTest extends CodeGenerator_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("com.google.javascript.jscomp.CodePrinter$PrettyCodePrinter");
      String string0 = compiler0.toSource(node0);
      assertEquals("com.google.javascript.jscomp.CodePrinter$PrettyCodePrinter", string0);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      CodeGenerator codeGenerator0 = CodeGenerator.forCostEstimation((CodeConsumer) null);
      // Undeclared exception!
      try { 
        codeGenerator0.tagAsStrict();
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.CodeGenerator", e);
      }
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      CodeGenerator codeGenerator0 = CodeGenerator.forCostEstimation((CodeConsumer) null);
      // Undeclared exception!
      try { 
        codeGenerator0.addCaseBody((Node) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.CodeGenerator", e);
      }
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      CompilerOptions compilerOptions0 = new CompilerOptions();
      CodeGenerator codeGenerator0 = new CodeGenerator((CodeConsumer) null, compilerOptions0);
      Charset charset0 = Charset.defaultCharset();
      CharsetEncoder charsetEncoder0 = charset0.newEncoder();
      String string0 = codeGenerator0.regexpEscape("\u0000\u00AD\u0600jd\u1680\u180En\u2028\u205F\u206A\u3000P\uD800\uFEFF\uFFF9\uFFFA", charsetEncoder0);
      assertEquals("/\\x00\u00AD\u0600jd\u1680\u180En\\u2028\u205F\u206A\u3000P\\ud800\uFEFF\uFFF9\uFFFA/", string0);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      CodeGenerator codeGenerator0 = CodeGenerator.forCostEstimation((CodeConsumer) null);
      codeGenerator0.addList((Node) null);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      CodeGenerator codeGenerator0 = CodeGenerator.forCostEstimation((CodeConsumer) null);
      codeGenerator0.addList((Node) null, false);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      LightweightMessageFormatter lightweightMessageFormatter0 = LightweightMessageFormatter.withoutSource();
      Logger logger0 = Logger.getGlobal();
      LoggerErrorManager loggerErrorManager0 = new LoggerErrorManager(lightweightMessageFormatter0, logger0);
      Compiler compiler0 = new Compiler(loggerErrorManager0);
      Node node0 = Normalize.parseAndNormalizeTestCode(compiler0, "6OBE[");
      Node node1 = new Node(51, node0, node0, node0, 2, (-1219));
      // Undeclared exception!
      try { 
        compiler0.toSource(node1);
        fail("Expecting exception: Error");
      
      } catch(Error e) {
         //
         // Unknown type 51
         // IN
         //     SCRIPT [source_file: com.google.javascript.rhino.Node$ObjectPropListItem@0000000730] [input_id: com.google.javascript.rhino.Node$ObjectPropListItem@0000000731]
         //
         verifyException("com.google.javascript.jscomp.CodeGenerator", e);
      }
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("com.google.javascript.jscomp.CodePrinter$PrettyCodePrinter");
      Node node1 = new Node(16, node0, node0, node0, 32, 1594);
      // Undeclared exception!
      try { 
        compiler0.toSource(node1);
        fail("Expecting exception: Error");
      
      } catch(Error e) {
         //
         // Unknown type 16
         // GT 32
         //     SCRIPT 1 [synthetic: com.google.javascript.rhino.Node$IntPropListItem@0000000753] [source_file: com.google.javascript.rhino.Node$ObjectPropListItem@0000000754] [input_id: com.google.javascript.rhino.Node$ObjectPropListItem@0000000755]
         //         EXPR_RESULT 1 [source_file: com.google.javascript.rhino.Node$ObjectPropListItem@0000000732]
         //             GETPROP 1 [source_file: com.google.javascript.rhino.Node$ObjectPropListItem@0000000732]
         //                 GETPROP 1 [source_file: com.google.javascript.rhino.Node$ObjectPropListItem@0000000732]
         //                     GETPROP 1 [source_file: com.google.javascript.rhino.Node$ObjectPropListItem@0000000732]
         //                         GETPROP 1 [source_file: com.google.javascript.rhino.Node$ObjectPropListItem@0000000732]
         //                             NAME com 1 [source_file: com.google.javascript.rhino.Node$ObjectPropListItem@0000000732]
         //                             STRING google 1 [source_file: com.google.javascript.rhino.Node$ObjectPropListItem@0000000732]
         //                         STRING javascript 1 [source_file: com.google.javascript.rhino.Node$ObjectPropListItem@0000000732]
         //                     STRING jscomp 1 [source_file: com.google.javascript.rhino.Node$ObjectPropListItem@0000000732]
         //                 STRING CodePrinter$PrettyCodePrinter 1 [source_file: com.google.javascript.rhino.Node$ObjectPropListItem@0000000732]
         //
         verifyException("com.google.javascript.jscomp.CodeGenerator", e);
      }
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      LightweightMessageFormatter lightweightMessageFormatter0 = LightweightMessageFormatter.withoutSource();
      Logger logger0 = Logger.getGlobal();
      LoggerErrorManager loggerErrorManager0 = new LoggerErrorManager(lightweightMessageFormatter0, logger0);
      Compiler compiler0 = new Compiler(loggerErrorManager0);
      Node node0 = Normalize.parseAndNormalizeTestCode(compiler0, "6OBE[");
      Node node1 = new Node(18, node0, node0, node0, (-473), (-1219));
      // Undeclared exception!
      try { 
        compiler0.toSource(node1);
        fail("Expecting exception: Error");
      
      } catch(Error e) {
         //
         // Unknown type 18
         // LSH
         //     SCRIPT [source_file: com.google.javascript.rhino.Node$ObjectPropListItem@0000000730] [input_id: com.google.javascript.rhino.Node$ObjectPropListItem@0000000731]
         //
         verifyException("com.google.javascript.jscomp.CodeGenerator", e);
      }
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = Normalize.parseAndNormalizeTestCode(compiler0, "com.google.javascript.jscomp.CodePrinter$PrettyCodePrinter");
      Node node1 = new Node(21, node0, node0, node0, (-473), 30);
      // Undeclared exception!
      try { 
        compiler0.toSource(node1);
        fail("Expecting exception: Error");
      
      } catch(Error e) {
         //
         // Unknown type 21
         // ADD
         //     SCRIPT 1 [synthetic: com.google.javascript.rhino.Node$IntPropListItem@0000000753] [source_file: com.google.javascript.rhino.Node$ObjectPropListItem@0000000754] [input_id: com.google.javascript.rhino.Node$ObjectPropListItem@0000000755]
         //         EXPR_RESULT 1 [source_file: com.google.javascript.rhino.Node$ObjectPropListItem@0000000732]
         //             GETPROP 1 [source_file: com.google.javascript.rhino.Node$ObjectPropListItem@0000000732]
         //                 GETPROP 1 [source_file: com.google.javascript.rhino.Node$ObjectPropListItem@0000000732]
         //                     GETPROP 1 [source_file: com.google.javascript.rhino.Node$ObjectPropListItem@0000000732]
         //                         GETPROP 1 [source_file: com.google.javascript.rhino.Node$ObjectPropListItem@0000000732]
         //                             NAME com 1 [source_file: com.google.javascript.rhino.Node$ObjectPropListItem@0000000732]
         //                             STRING google 1 [source_file: com.google.javascript.rhino.Node$ObjectPropListItem@0000000732]
         //                         STRING javascript 1 [source_file: com.google.javascript.rhino.Node$ObjectPropListItem@0000000732]
         //                     STRING jscomp 1 [source_file: com.google.javascript.rhino.Node$ObjectPropListItem@0000000732]
         //                 STRING CodePrinter$PrettyCodePrinter 1 [source_file: com.google.javascript.rhino.Node$ObjectPropListItem@0000000732]
         //
         verifyException("com.google.javascript.jscomp.CodeGenerator", e);
      }
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      LightweightMessageFormatter lightweightMessageFormatter0 = LightweightMessageFormatter.withoutSource();
      Logger logger0 = Logger.getGlobal();
      LoggerErrorManager loggerErrorManager0 = new LoggerErrorManager(lightweightMessageFormatter0, logger0);
      Compiler compiler0 = new Compiler(loggerErrorManager0);
      Node node0 = Normalize.parseAndNormalizeTestCode(compiler0, "2");
      String string0 = compiler0.toSource(node0);
      assertEquals("2", string0);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      LightweightMessageFormatter lightweightMessageFormatter0 = LightweightMessageFormatter.withoutSource();
      Logger logger0 = Logger.getGlobal();
      LoggerErrorManager loggerErrorManager0 = new LoggerErrorManager(lightweightMessageFormatter0, logger0);
      Compiler compiler0 = new Compiler(loggerErrorManager0);
      Node node0 = Normalize.parseAndNormalizeTestCode(compiler0, "2");
      Node node1 = new Node(48, node0, node0, node0, 56, 0);
      // Undeclared exception!
      try { 
        compiler0.toSource(node1);
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // 48
         //
         verifyException("com.google.javascript.rhino.Token", e);
      }
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("gkJ");
      Node node1 = new Node(49, node0, node0, node0, 16, 57);
      String string0 = compiler0.toSource(node1);
      assertEquals("throw gkJ;", string0);
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      MockPrintStream mockPrintStream0 = new MockPrintStream("!--gyoa");
      Compiler compiler0 = new Compiler(mockPrintStream0);
      Node node0 = new Node(52);
      // Undeclared exception!
      try { 
        compiler0.toSource(node0);
        fail("Expecting exception: Error");
      
      } catch(Error e) {
         //
         // Unknown type 52
         // INSTANCEOF
         //
         verifyException("com.google.javascript.jscomp.CodeGenerator", e);
      }
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("com.google.javascript.jscomp.CodePrinter$PrettyCodePrinter");
      Node node1 = new Node(136, node0, node0, node0, 32, 4);
      // Undeclared exception!
      try { 
        compiler0.toSource(node1);
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // 136
         //
         verifyException("com.google.javascript.rhino.Token", e);
      }
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("gkJ");
      node0.setType((-1057));
      // Undeclared exception!
      try { 
        compiler0.toSource(node0);
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // -1057
         //
         verifyException("com.google.javascript.rhino.Token", e);
      }
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      double double0 = CodeGenerator.getSimpleNumber("2");
      assertEquals(2.0, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      double double0 = CodeGenerator.getSimpleNumber("");
      assertEquals(Double.NaN, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      double double0 = CodeGenerator.getSimpleNumber("!--");
      assertEquals(Double.NaN, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      double double0 = CodeGenerator.getSimpleNumber("com.google.common.baDe.CharMatcher$Or");
      assertEquals(Double.NaN, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      CodeGenerator codeGenerator0 = CodeGenerator.forCostEstimation((CodeConsumer) null);
      Node node0 = new Node(86, 86, 86);
      CodeGenerator.Context codeGenerator_Context0 = CodeGenerator.Context.IN_FOR_INIT_CLAUSE;
      // Undeclared exception!
      try { 
        codeGenerator0.addList(node0, false, codeGenerator_Context0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.CodeGenerator", e);
      }
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      CodeGenerator codeGenerator0 = CodeGenerator.forCostEstimation((CodeConsumer) null);
      Node node0 = new Node(86);
      CodeGenerator.Context codeGenerator_Context0 = CodeGenerator.Context.PRESERVE_BLOCK;
      // Undeclared exception!
      try { 
        codeGenerator0.addList(node0, true, codeGenerator_Context0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.CodeGenerator", e);
      }
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      CodeGenerator codeGenerator0 = CodeGenerator.forCostEstimation((CodeConsumer) null);
      Node node0 = Node.newString("XFd");
      // Undeclared exception!
      try { 
        codeGenerator0.addArrayList(node0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.CodeGenerator", e);
      }
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      CodeGenerator codeGenerator0 = CodeGenerator.forCostEstimation((CodeConsumer) null);
      codeGenerator0.addAllSiblings((Node) null);
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      Node node0 = Node.newString("p3");
      CodeGenerator codeGenerator0 = CodeGenerator.forCostEstimation((CodeConsumer) null);
      // Undeclared exception!
      try { 
        codeGenerator0.addAllSiblings(node0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.CodeGenerator", e);
      }
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      CodeGenerator codeGenerator0 = CodeGenerator.forCostEstimation((CodeConsumer) null);
      String string0 = codeGenerator0.escapeToDoubleQuotedJsString("\n}");
      assertEquals("\"\\n}\"", string0);
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      CodeGenerator codeGenerator0 = CodeGenerator.forCostEstimation((CodeConsumer) null);
      String string0 = codeGenerator0.regexpEscape(",,4'y?nALZJbRJ");
      assertEquals("/,,4'y?nALZJbRJ/", string0);
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      CompilerOptions compilerOptions0 = compiler0.newCompilerOptions();
      CodeGenerator codeGenerator0 = new CodeGenerator((CodeConsumer) null, compilerOptions0);
      String string0 = codeGenerator0.regexpEscape("um>9IB");
      assertEquals("/um>9IB/", string0);
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      CodeGenerator codeGenerator0 = CodeGenerator.forCostEstimation((CodeConsumer) null);
      String string0 = codeGenerator0.escapeToDoubleQuotedJsString("y2*M>=90-P\\mLFq");
      assertEquals("\"y2\\u007f*M>=90-P\\\\mLFq\"", string0);
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      CompilerOptions compilerOptions0 = new CompilerOptions();
      CodeGenerator codeGenerator0 = new CodeGenerator((CodeConsumer) null, compilerOptions0);
      String string0 = codeGenerator0.regexpEscape("JI3?lEv*:=20\"[b$I");
      assertEquals("/JI3?lEv*:=20\"[b$I/", string0);
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      CodeGenerator codeGenerator0 = CodeGenerator.forCostEstimation((CodeConsumer) null);
      Charset charset0 = Charset.defaultCharset();
      CharsetEncoder charsetEncoder0 = charset0.newEncoder();
      String string0 = codeGenerator0.regexpEscape("MB&", charsetEncoder0);
      assertEquals("/MB&/", string0);
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      CompilerOptions compilerOptions0 = compiler0.newCompilerOptions();
      CodeGenerator codeGenerator0 = new CodeGenerator((CodeConsumer) null, compilerOptions0);
      String string0 = codeGenerator0.regexpEscape("r&\"jD%V");
      assertEquals("/r&\"jD%V/", string0);
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      CompilerOptions compilerOptions0 = new CompilerOptions();
      CodeGenerator codeGenerator0 = new CodeGenerator((CodeConsumer) null, compilerOptions0);
      String string0 = codeGenerator0.escapeToDoubleQuotedJsString("u>\"*8<!--K((~!V5(");
      assertEquals("\"u\\x3e\\\"*8\\x3c!--K((~!V5(\"", string0);
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      CodeGenerator codeGenerator0 = CodeGenerator.forCostEstimation((CodeConsumer) null);
      String string0 = codeGenerator0.regexpEscape("e>w}Pb`(E t3");
      assertEquals("/e>w}Pb`(E t3/", string0);
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      CodeGenerator codeGenerator0 = CodeGenerator.forCostEstimation((CodeConsumer) null);
      String string0 = codeGenerator0.escapeToDoubleQuotedJsString("/)ax</script{e,o5sf/");
      assertEquals("\"/)ax\\x3c/script{e,o5s\\u007ff/\"", string0);
  }

  @Test(timeout = 4000)
  public void test35()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      CompilerOptions compilerOptions0 = compiler0.newCompilerOptions();
      CodeGenerator codeGenerator0 = new CodeGenerator((CodeConsumer) null, compilerOptions0);
      String string0 = codeGenerator0.regexpEscape("kI+]<!--7");
      assertEquals("/kI+]\\x3c!--7/", string0);
  }

  @Test(timeout = 4000)
  public void test36()  throws Throwable  {
      CodeGenerator codeGenerator0 = CodeGenerator.forCostEstimation((CodeConsumer) null);
      String string0 = codeGenerator0.escapeToDoubleQuotedJsString(")ax<XFds/G\"{e,o5sF");
      assertEquals("\")ax<XFds/G\\\"{e,o5s\\u007fF\"", string0);
  }

  @Test(timeout = 4000)
  public void test37()  throws Throwable  {
      CompilerOptions compilerOptions0 = new CompilerOptions();
      CodeGenerator codeGenerator0 = new CodeGenerator((CodeConsumer) null, compilerOptions0);
      String string0 = codeGenerator0.escapeToDoubleQuotedJsString(":=MlRT.[\u0004lJ&");
      assertEquals("\":\\x3dMlRT.[\\u0004lJ\\x26\"", string0);
  }

  @Test(timeout = 4000)
  public void test38()  throws Throwable  {
      String string0 = CodeGenerator.identifierEscape("\u0000\u00AD\u0600Gd\u1680\u180En\u2028\u205F\u206A\u3000P\uD800\uFEFF\uFFF9\uFFFA");
      assertEquals("\\u0000\\u007f\\u00ad\\u0600Gd\\u1680\\u180en\\u2028\\u205f\\u206a\\u3000P\\ud800\\ufeff\\ufff9\\ufffa", string0);
  }
}