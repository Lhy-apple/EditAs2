/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 13:19:36 GMT 2023
 */

package com.google.javascript.jscomp;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.google.common.base.Supplier;
import com.google.javascript.jscomp.AbstractCompiler;
import com.google.javascript.jscomp.Compiler;
import com.google.javascript.jscomp.FunctionInjector;
import com.google.javascript.jscomp.InlineFunctions;
import com.google.javascript.jscomp.JSModule;
import com.google.javascript.jscomp.Normalize;
import com.google.javascript.jscomp.SimpleFunctionAliasAnalysis;
import com.google.javascript.jscomp.SpecializeModule;
import com.google.javascript.rhino.Node;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.mock.java.io.MockPrintStream;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class InlineFunctions_ESTest extends InlineFunctions_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Supplier<String> supplier0 = compiler0.getUniqueNameIdSupplier();
      InlineFunctions inlineFunctions0 = new InlineFunctions(compiler0, supplier0, true, true, true);
      SimpleFunctionAliasAnalysis simpleFunctionAliasAnalysis0 = new SimpleFunctionAliasAnalysis();
      SpecializeModule.SpecializationState specializeModule_SpecializationState0 = new SpecializeModule.SpecializationState(simpleFunctionAliasAnalysis0);
      inlineFunctions0.enableSpecialization(specializeModule_SpecializationState0);
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      MockPrintStream mockPrintStream0 = new MockPrintStream("a0Vb?D5\"6:M9_uJIm");
      Compiler compiler0 = new Compiler(mockPrintStream0);
      Supplier<String> supplier0 = compiler0.getUniqueNameIdSupplier();
      InlineFunctions inlineFunctions0 = new InlineFunctions(compiler0, supplier0, false, false, false);
      Node node0 = Normalize.parseAndNormalizeTestCode(compiler0, "a0Vb?D5\"6:M9_uJIm", "a0Vb?D5\"6:M9_uJIm");
      FunctionInjector.InliningMode functionInjector_InliningMode0 = FunctionInjector.InliningMode.BLOCK;
      InlineFunctions.Reference inlineFunctions_Reference0 = inlineFunctions0.new Reference(node0, (JSModule) null, functionInjector_InliningMode0, false);
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      InlineFunctions inlineFunctions0 = null;
      try {
        inlineFunctions0 = new InlineFunctions((AbstractCompiler) null, (Supplier<String>) null, false, false, false);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.common.base.Preconditions", e);
      }
  }

  @Test(timeout = 4000)
  public void test3()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      InlineFunctions inlineFunctions0 = null;
      try {
        inlineFunctions0 = new InlineFunctions(compiler0, (Supplier<String>) null, true, true, true);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.common.base.Preconditions", e);
      }
  }

  @Test(timeout = 4000)
  public void test4()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      AbstractCompiler.LifeCycleStage abstractCompiler_LifeCycleStage0 = AbstractCompiler.LifeCycleStage.NORMALIZED;
      compiler0.setLifeCycleStage(abstractCompiler_LifeCycleStage0);
      Node node0 = compiler0.parseTestCode("com.google.javascript.jscomp.InlineFunctions$FunctionExpression");
      Supplier<String> supplier0 = compiler0.getUniqueNameIdSupplier();
      InlineFunctions inlineFunctions0 = new InlineFunctions(compiler0, supplier0, false, false, false);
      inlineFunctions0.process(node0, node0);
      assertEquals(0, Node.SIDE_EFFECTS_ALL);
  }

  @Test(timeout = 4000)
  public void test5()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      AbstractCompiler.LifeCycleStage abstractCompiler_LifeCycleStage0 = AbstractCompiler.LifeCycleStage.NORMALIZED;
      compiler0.setLifeCycleStage(abstractCompiler_LifeCycleStage0);
      Node node0 = compiler0.parseTestCode("com.google.javascript.jscomp.InlineFunctions$FunctionExpression");
      Supplier<String> supplier0 = compiler0.getUniqueNameIdSupplier();
      InlineFunctions inlineFunctions0 = new InlineFunctions(compiler0, supplier0, true, true, true);
      // Undeclared exception!
      try { 
        inlineFunctions0.process(node0, node0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // INTERNAL COMPILER ERROR.
         // Please report this problem.
         // null
         //   Node(SCRIPT):  [testcode] :1:0
         // [source unknown]
         //   Parent: NULL
         //
         verifyException("com.google.common.base.Preconditions", e);
      }
  }

  @Test(timeout = 4000)
  public void test6()  throws Throwable  {
      MockPrintStream mockPrintStream0 = new MockPrintStream("a0Vb?D5\"6:M9_uJIm");
      Compiler compiler0 = new Compiler(mockPrintStream0);
      Node node0 = Normalize.parseAndNormalizeTestCode(compiler0, "a0Vb?D5\"6:M9_uJIm", "a0Vb?D5\"6:M9_uJIm");
      // Undeclared exception!
      try { 
        InlineFunctions.isCandidateUsage(node0);
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.common.base.Preconditions", e);
      }
  }

  @Test(timeout = 4000)
  public void test7()  throws Throwable  {
      Node node0 = new Node(38);
      // Undeclared exception!
      try { 
        InlineFunctions.isCandidateUsage(node0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.InlineFunctions", e);
      }
  }

  @Test(timeout = 4000)
  public void test8()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Supplier<String> supplier0 = compiler0.getUniqueNameIdSupplier();
      InlineFunctions inlineFunctions0 = new InlineFunctions(compiler0, supplier0, false, false, false);
      inlineFunctions0.trimCanidatesUsingOnCost();
  }

  @Test(timeout = 4000)
  public void test9()  throws Throwable  {
      MockPrintStream mockPrintStream0 = new MockPrintStream("a0Vb?D5\"6:M9_uJIm");
      Compiler compiler0 = new Compiler(mockPrintStream0);
      Supplier<String> supplier0 = compiler0.getUniqueNameIdSupplier();
      InlineFunctions inlineFunctions0 = new InlineFunctions(compiler0, supplier0, false, false, false);
      inlineFunctions0.removeInlinedFunctions();
  }
}