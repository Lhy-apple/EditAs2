/*
 * This file was automatically generated by EvoSuite
 * Sat Jul 29 18:27:20 GMT 2023
 */

package com.google.javascript.jscomp;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.google.javascript.jscomp.Compiler;
import com.google.javascript.jscomp.JSError;
import com.google.javascript.jscomp.JsAst;
import com.google.javascript.jscomp.LightweightMessageFormatter;
import com.google.javascript.jscomp.SourceFile;
import com.google.javascript.rhino.Node;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.mock.java.io.MockPrintStream;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class JsAst_ESTest extends JsAst_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      // Undeclared exception!
      try { 
        compiler0.parseSyntheticCode("JSC_INVALID_TWEAK_ID_ERROR");
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.Compiler", e);
      }
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      SourceFile sourceFile0 = SourceFile.fromFile(";}W1z");
      JsAst jsAst0 = new JsAst(sourceFile0);
      jsAst0.clearAst();
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      SourceFile sourceFile0 = SourceFile.fromFile(";}W1z");
      JsAst jsAst0 = new JsAst(sourceFile0);
      jsAst0.setSourceFile(sourceFile0);
      assertEquals(";}W1z", sourceFile0.getName());
  }

  @Test(timeout = 4000)
  public void test3()  throws Throwable  {
      SourceFile sourceFile0 = SourceFile.fromFile(";}W1z");
      Compiler compiler0 = new Compiler();
      compiler0.initCompilerOptionsIfTesting();
      JsAst jsAst0 = new JsAst(sourceFile0);
      jsAst0.getAstRoot(compiler0);
      jsAst0.getAstRoot(compiler0);
      assertTrue(compiler0.hasErrors());
      assertEquals(1, compiler0.getErrorCount());
  }

  @Test(timeout = 4000)
  public void test4()  throws Throwable  {
      SourceFile sourceFile0 = SourceFile.fromFile(";}W1z");
      LightweightMessageFormatter.withoutSource();
      Compiler compiler0 = new Compiler();
      compiler0.initCompilerOptionsIfTesting();
      String[] stringArray0 = new String[8];
      JSError jSError0 = JSError.make(compiler0.MOTION_ITERATIONS_ERROR, stringArray0);
      compiler0.report(jSError0);
      JsAst jsAst0 = new JsAst(sourceFile0);
      MockPrintStream mockPrintStream0 = new MockPrintStream(";}W1z");
      Node node0 = jsAst0.getAstRoot(compiler0);
      assertNotNull(node0);
      assertEquals((-1), node0.getSourcePosition());
  }

  @Test(timeout = 4000)
  public void test5()  throws Throwable  {
      SourceFile sourceFile0 = SourceFile.fromFile(";}W1z");
      LightweightMessageFormatter.withoutSource();
      Compiler compiler0 = new Compiler();
      compiler0.initCompilerOptionsIfTesting();
      MockPrintStream mockPrintStream0 = new MockPrintStream(";}W1z");
      Compiler compiler1 = new Compiler(mockPrintStream0);
      JsAst jsAst0 = new JsAst(sourceFile0);
      // Undeclared exception!
      try { 
        jsAst0.getAstRoot(compiler0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // duplicate key: desc
         //
         verifyException("com.google.common.base.Preconditions", e);
      }
  }
}