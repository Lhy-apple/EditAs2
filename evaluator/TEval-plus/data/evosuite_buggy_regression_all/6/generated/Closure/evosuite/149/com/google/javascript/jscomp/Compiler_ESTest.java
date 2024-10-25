/*
 * This file was automatically generated by EvoSuite
 * Wed Sep 27 00:06:17 GMT 2023
 */

package com.google.javascript.jscomp;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.google.common.base.Supplier;
import com.google.javascript.jscomp.CodeChangeHandler;
import com.google.javascript.jscomp.Compiler;
import com.google.javascript.jscomp.CompilerInput;
import com.google.javascript.jscomp.CompilerOptions;
import com.google.javascript.jscomp.JSModule;
import com.google.javascript.jscomp.JSModuleGraph;
import com.google.javascript.jscomp.JSSourceFile;
import com.google.javascript.jscomp.JsAst;
import com.google.javascript.jscomp.LightweightMessageFormatter;
import com.google.javascript.jscomp.LoggerErrorManager;
import com.google.javascript.jscomp.Region;
import com.google.javascript.jscomp.ReverseAbstractInterpreter;
import com.google.javascript.jscomp.SourceMap;
import com.google.javascript.jscomp.SyntheticAst;
import com.google.javascript.jscomp.Tracer;
import com.google.javascript.jscomp.VariableMap;
import com.google.javascript.rhino.Node;
import java.io.ByteArrayOutputStream;
import java.io.DataOutputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.Stack;
import java.util.logging.Logger;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.mock.java.io.MockFile;
import org.evosuite.runtime.mock.java.io.MockPrintStream;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class Compiler_ESTest extends Compiler_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      // Undeclared exception!
      try { 
        compiler0.toSource((JSModule) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.Compiler", e);
      }
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      // Undeclared exception!
      try { 
        compiler0.toSourceArray();
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.Compiler", e);
      }
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      LightweightMessageFormatter lightweightMessageFormatter0 = LightweightMessageFormatter.withoutSource();
      Logger logger0 = Logger.getGlobal();
      LoggerErrorManager loggerErrorManager0 = new LoggerErrorManager(lightweightMessageFormatter0, logger0);
      Compiler compiler0 = new Compiler(loggerErrorManager0);
      // Undeclared exception!
      try { 
        compiler0.toSource();
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.Compiler", e);
      }
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Compiler.CodeBuilder compiler_CodeBuilder0 = new Compiler.CodeBuilder();
      // Undeclared exception!
      try { 
        compiler0.toSource(compiler_CodeBuilder0, (-3258), (Node) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.Compiler", e);
      }
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      // Undeclared exception!
      try { 
        compiler0.normalize();
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.Compiler", e);
      }
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      MockPrintStream mockPrintStream0 = new MockPrintStream("TR#X^#{^!Ni-T");
      Compiler compiler0 = new Compiler(mockPrintStream0);
      JSSourceFile jSSourceFile0 = JSSourceFile.fromFile("TR#X^#{^!Ni-T");
      JSSourceFile[] jSSourceFileArray0 = new JSSourceFile[3];
      jSSourceFileArray0[0] = jSSourceFile0;
      jSSourceFileArray0[1] = jSSourceFile0;
      jSSourceFileArray0[2] = jSSourceFile0;
      CompilerOptions compilerOptions0 = new CompilerOptions();
      compiler0.compile(jSSourceFile0, jSSourceFileArray0, compilerOptions0);
      JsAst jsAst0 = new JsAst(jSSourceFile0);
      // Undeclared exception!
      try { 
        compiler0.addIncrementalSourceAst(jsAst0);
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // Duplicate input of name TR#X^#{^!Ni-T
         //
         verifyException("com.google.common.base.Preconditions", e);
      }
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      LightweightMessageFormatter lightweightMessageFormatter0 = LightweightMessageFormatter.withoutSource();
      Logger logger0 = Logger.getAnonymousLogger();
      LoggerErrorManager loggerErrorManager0 = new LoggerErrorManager(lightweightMessageFormatter0, logger0);
      Compiler compiler0 = new Compiler(loggerErrorManager0);
      Node node0 = Node.newString(160, "%name%");
      // Undeclared exception!
      try { 
        compiler0.toSource(node0);
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // 160
         //
         verifyException("com.google.javascript.rhino.Token", e);
      }
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      MockPrintStream mockPrintStream0 = new MockPrintStream("TR#X^#{^!Ni-T");
      Compiler compiler0 = new Compiler(mockPrintStream0);
      boolean boolean0 = compiler0.hasRegExpGlobalReferences();
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      ByteArrayOutputStream byteArrayOutputStream0 = new ByteArrayOutputStream(1656);
      DataOutputStream dataOutputStream0 = new DataOutputStream(byteArrayOutputStream0);
      MockPrintStream mockPrintStream0 = new MockPrintStream(dataOutputStream0, true);
      Compiler compiler0 = new Compiler(mockPrintStream0);
      Compiler.IntermediateState compiler_IntermediateState0 = compiler0.getState();
      compiler0.setState(compiler_IntermediateState0);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      SyntheticAst syntheticAst0 = new SyntheticAst("");
      Node node0 = syntheticAst0.getAstRoot(compiler0);
      // Undeclared exception!
      try { 
        compiler0.prepareAst(node0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.Compiler", e);
      }
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      compiler0.disableThreads();
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      MockPrintStream mockPrintStream0 = new MockPrintStream("TR#X^#{^!Ni-T");
      Compiler compiler0 = new Compiler(mockPrintStream0);
      JSSourceFile jSSourceFile0 = JSSourceFile.fromFile("TR#X^#{^!Ni-T");
      JsAst jsAst0 = new JsAst(jSSourceFile0);
      // Undeclared exception!
      try { 
        compiler0.addIncrementalSourceAst(jsAst0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.Compiler", e);
      }
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      MockFile mockFile0 = new MockFile("RJKy?'TeC#hE-6hxF", "");
      MockPrintStream mockPrintStream0 = new MockPrintStream(mockFile0);
      Compiler compiler0 = new Compiler(mockPrintStream0);
      boolean boolean0 = compiler0.isNormalized();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      MockPrintStream mockPrintStream0 = new MockPrintStream("TR#X^#{^!Ni-T");
      Compiler compiler0 = new Compiler(mockPrintStream0);
      compiler0.setUnnormalized();
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      Logger logger0 = Logger.getAnonymousLogger();
      LoggerErrorManager loggerErrorManager0 = new LoggerErrorManager(logger0);
      Compiler compiler0 = new Compiler(loggerErrorManager0);
      int int0 = compiler0.getWarningCount();
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      MockPrintStream mockPrintStream0 = new MockPrintStream("TR#X^#{^!Ni-T");
      Compiler compiler0 = new Compiler(mockPrintStream0);
      VariableMap variableMap0 = compiler0.getVariableMap();
      assertNull(variableMap0);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      // Undeclared exception!
      try { 
        compiler0.parse();
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.Compiler", e);
      }
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      Logger logger0 = Logger.getAnonymousLogger();
      LoggerErrorManager loggerErrorManager0 = new LoggerErrorManager(logger0);
      Compiler compiler0 = new Compiler(loggerErrorManager0);
      Supplier<String> supplier0 = compiler0.getUniqueNameIdSupplier();
      assertNotNull(supplier0);
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      Logger logger0 = Tracer.logger;
      LoggerErrorManager loggerErrorManager0 = new LoggerErrorManager(logger0);
      Compiler compiler0 = new Compiler(loggerErrorManager0);
      // Undeclared exception!
      try { 
        compiler0.recordFunctionInformation();
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.Compiler", e);
      }
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      Logger logger0 = Logger.getLogger("f&/;I(Ig@ldDf,I");
      LoggerErrorManager loggerErrorManager0 = new LoggerErrorManager(logger0);
      Compiler compiler0 = new Compiler(loggerErrorManager0);
      JSModuleGraph jSModuleGraph0 = compiler0.getModuleGraph();
      assertNull(jSModuleGraph0);
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      ByteArrayOutputStream byteArrayOutputStream0 = new ByteArrayOutputStream(1656);
      DataOutputStream dataOutputStream0 = new DataOutputStream(byteArrayOutputStream0);
      MockPrintStream mockPrintStream0 = new MockPrintStream(dataOutputStream0, true);
      Compiler compiler0 = new Compiler(mockPrintStream0);
      SourceMap sourceMap0 = compiler0.getSourceMap();
      assertNull(sourceMap0);
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      MockPrintStream mockPrintStream0 = new MockPrintStream("TR#X^#{^!Ni-T");
      Compiler compiler0 = new Compiler(mockPrintStream0);
      // Undeclared exception!
      try { 
        compiler0.isTypeCheckingEnabled();
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.Compiler", e);
      }
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      compiler0.removeChangeHandler((CodeChangeHandler) null);
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      compiler0.getErrorManager();
      // Undeclared exception!
      try { 
        compiler0.removeTryCatchFinally();
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // INTERNAL COMPILER ERROR.
         // Please report this problem.
         // null
         //
         verifyException("com.google.common.base.Preconditions", e);
      }
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      compiler0.setHasRegExpGlobalReferences(false);
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      ByteArrayOutputStream byteArrayOutputStream0 = new ByteArrayOutputStream(1656);
      DataOutputStream dataOutputStream0 = new DataOutputStream(byteArrayOutputStream0);
      MockPrintStream mockPrintStream0 = new MockPrintStream(dataOutputStream0, true);
      Compiler compiler0 = new Compiler(mockPrintStream0);
      // Undeclared exception!
      try { 
        compiler0.getMessages();
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.Compiler", e);
      }
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      MockPrintStream mockPrintStream0 = new MockPrintStream("com.google.javascript.jscomp.CrossModuleCodeMotion");
      Compiler compiler0 = new Compiler(mockPrintStream0);
      List<CompilerInput> list0 = compiler0.getExternsForTesting();
      assertNull(list0);
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      compiler0.addToDebugLog("com.google.javascript.jscomp.mozilla.rhino.BaseFunction");
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      Compiler.CodeBuilder compiler_CodeBuilder0 = new Compiler.CodeBuilder();
      int int0 = compiler_CodeBuilder0.getLength();
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      Compiler.CodeBuilder compiler_CodeBuilder0 = new Compiler.CodeBuilder();
      String string0 = compiler_CodeBuilder0.toString();
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      Compiler.CodeBuilder compiler_CodeBuilder0 = new Compiler.CodeBuilder();
      int int0 = compiler_CodeBuilder0.getLineIndex();
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      Compiler.CodeBuilder compiler_CodeBuilder0 = new Compiler.CodeBuilder();
      compiler_CodeBuilder0.reset();
      assertEquals("", compiler_CodeBuilder0.toString());
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      MockPrintStream mockPrintStream0 = new MockPrintStream("TR#X^#{^!Ni-T");
      Compiler compiler0 = new Compiler(mockPrintStream0);
      CompilerOptions compilerOptions0 = new CompilerOptions();
      Stack<JSSourceFile> stack0 = new Stack<JSSourceFile>();
      ArrayList<JSModule> arrayList0 = new ArrayList<JSModule>();
      compiler0.initModules(stack0, arrayList0, compilerOptions0);
      assertEquals(1, compiler0.getErrorCount());
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      // Undeclared exception!
      try { 
        compiler0.endPass();
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // Tracer should not be null at the end of a pass.
         //
         verifyException("com.google.common.base.Preconditions", e);
      }
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      // Undeclared exception!
      try { 
        compiler0.getTypeValidator();
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.Compiler", e);
      }
  }

  @Test(timeout = 4000)
  public void test35()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      compiler0.getErrorManager();
      ReverseAbstractInterpreter reverseAbstractInterpreter0 = compiler0.getReverseAbstractInterpreter();
      assertNotNull(reverseAbstractInterpreter0);
  }

  @Test(timeout = 4000)
  public void test36()  throws Throwable  {
      Compiler.CodeBuilder compiler_CodeBuilder0 = new Compiler.CodeBuilder();
      Compiler.CodeBuilder compiler_CodeBuilder1 = compiler_CodeBuilder0.append("com.google.javascript.jscomp.Compiler$4");
      assertEquals("com.google.javascript.jscomp.Compiler$4", compiler_CodeBuilder1.toString());
  }

  @Test(timeout = 4000)
  public void test37()  throws Throwable  {
      Compiler.CodeBuilder compiler_CodeBuilder0 = new Compiler.CodeBuilder();
      boolean boolean0 = compiler_CodeBuilder0.endsWith("TR#X^#{^!Ni-T");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test38()  throws Throwable  {
      MockPrintStream mockPrintStream0 = new MockPrintStream("com.google.javascript.jsomp.CrossModleCodeMotion");
      Compiler compiler0 = new Compiler(mockPrintStream0);
      compiler0.reportCodeChange();
  }

  @Test(timeout = 4000)
  public void test39()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      // Undeclared exception!
      try { 
        compiler0.getParserConfig();
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.Compiler", e);
      }
  }

  @Test(timeout = 4000)
  public void test40()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      // Undeclared exception!
      try { 
        compiler0.getSourceRegion("_c1_~|>gr5>'!J", 118);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
      }
  }

  @Test(timeout = 4000)
  public void test41()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Region region0 = compiler0.getSourceRegion(",-^[Vl5$hQV", (-3638));
      assertNull(region0);
  }

  @Test(timeout = 4000)
  public void test42()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      JSModule jSModule0 = new JSModule("Conflicting externs name: ");
      // Undeclared exception!
      try { 
        compiler0.getNodeForCodeInsertion(jSModule0);
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // Root module has no inputs
         //
         verifyException("com.google.javascript.jscomp.Compiler", e);
      }
  }

  @Test(timeout = 4000)
  public void test43()  throws Throwable  {
      MockPrintStream mockPrintStream0 = new MockPrintStream("TR#X^#{^!Ni-T");
      Compiler compiler0 = new Compiler(mockPrintStream0);
      String string0 = compiler0.getAstDotGraph();
      assertEquals("", string0);
  }
}
