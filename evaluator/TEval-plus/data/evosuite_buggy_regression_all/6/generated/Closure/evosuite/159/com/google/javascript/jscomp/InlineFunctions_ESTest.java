/*
 * This file was automatically generated by EvoSuite
 * Wed Sep 27 00:07:20 GMT 2023
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
import com.google.javascript.jscomp.SpecializeModule;
import com.google.javascript.rhino.Node;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class InlineFunctions_ESTest extends InlineFunctions_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Supplier<String> supplier0 = compiler0.getUniqueNameIdSupplier();
      InlineFunctions inlineFunctions0 = new InlineFunctions(compiler0, supplier0, false, false, false);
      inlineFunctions0.enableSpecialization((SpecializeModule.SpecializationState) null);
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Supplier<String> supplier0 = compiler0.getUniqueNameIdSupplier();
      InlineFunctions inlineFunctions0 = new InlineFunctions(compiler0, supplier0, true, false, false);
      JSModule jSModule0 = new JSModule("com.google.javascript.jscomp.DefaultPassConfig$16");
      FunctionInjector.InliningMode functionInjector_InliningMode0 = FunctionInjector.InliningMode.BLOCK;
      InlineFunctions.Reference inlineFunctions_Reference0 = inlineFunctions0.new Reference((Node) null, jSModule0, functionInjector_InliningMode0, false);
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Supplier<String> supplier0 = compiler0.getUniqueNameIdSupplier();
      InlineFunctions inlineFunctions0 = null;
      try {
        inlineFunctions0 = new InlineFunctions((AbstractCompiler) null, supplier0, true, true, true);
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
        inlineFunctions0 = new InlineFunctions(compiler0, (Supplier<String>) null, false, false, false);
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
      Supplier<String> supplier0 = compiler0.getUniqueNameIdSupplier();
      AbstractCompiler.LifeCycleStage abstractCompiler_LifeCycleStage0 = AbstractCompiler.LifeCycleStage.NORMALIZED_OBFUSCATED;
      compiler0.setLifeCycleStage(abstractCompiler_LifeCycleStage0);
      InlineFunctions inlineFunctions0 = new InlineFunctions(compiler0, supplier0, false, false, false);
      Node node0 = compiler0.parseSyntheticCode("function JSCompiler_returnArg(JSCompiler_returnArg_value) {  return function() {return JSCompiler_returnArg_value}}", "function JSCompiler_returnArg(JSCompiler_returnArg_value) {  return function() {return JSCompiler_returnArg_value}}");
      inlineFunctions0.process(node0, node0);
      assertEquals(49, Node.FREE_CALL);
  }

  @Test(timeout = 4000)
  public void test5()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      AbstractCompiler.LifeCycleStage abstractCompiler_LifeCycleStage0 = AbstractCompiler.LifeCycleStage.NORMALIZED;
      compiler0.setLifeCycleStage(abstractCompiler_LifeCycleStage0);
      Node node0 = compiler0.parseSyntheticCode("JSC_FRACTIONAL_BITWISE_OPERAND", "JSC_FRACTIONAL_BITWISE_OPERAND");
      Supplier<String> supplier0 = compiler0.getUniqueNameIdSupplier();
      InlineFunctions inlineFunctions0 = new InlineFunctions(compiler0, supplier0, true, true, true);
      // Undeclared exception!
      try { 
        inlineFunctions0.process(node0, node0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.Compiler", e);
      }
  }

  @Test(timeout = 4000)
  public void test6()  throws Throwable  {
      Node node0 = Node.newNumber((-3132.02786217));
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
      Node node0 = new Node(38, 38, 38);
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
      InlineFunctions inlineFunctions0 = new InlineFunctions(compiler0, supplier0, true, true, true);
      inlineFunctions0.trimCanidatesUsingOnCost();
  }

  @Test(timeout = 4000)
  public void test9()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Supplier<String> supplier0 = compiler0.getUniqueNameIdSupplier();
      InlineFunctions inlineFunctions0 = new InlineFunctions(compiler0, supplier0, true, true, true);
      inlineFunctions0.removeInlinedFunctions();
  }
}
