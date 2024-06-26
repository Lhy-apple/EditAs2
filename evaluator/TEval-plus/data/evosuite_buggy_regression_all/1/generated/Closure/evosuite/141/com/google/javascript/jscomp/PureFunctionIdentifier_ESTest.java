/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 13:18:17 GMT 2023
 */

package com.google.javascript.jscomp;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.google.javascript.jscomp.Compiler;
import com.google.javascript.jscomp.DefinitionProvider;
import com.google.javascript.jscomp.PureFunctionIdentifier;
import com.google.javascript.rhino.Node;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class PureFunctionIdentifier_ESTest extends PureFunctionIdentifier_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      PureFunctionIdentifier pureFunctionIdentifier0 = new PureFunctionIdentifier(compiler0, (DefinitionProvider) null);
      Node node0 = compiler0.parseTestCode("function JSCompiler_identityFn() {  return function(JSCompiler_identityFn_value) {return JSCompiler_identityFn_value}}");
      // Undeclared exception!
      try { 
        pureFunctionIdentifier0.process(node0, node0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // INTERNAL COMPILER ERROR.
         // Please report this problem.
         // null
         //   Node(FUNCTION JSCompiler_identityFn):  [testcode] :1:9
         // [source unknown]
         //   Parent(SCRIPT):  [testcode] :1:0
         // [source unknown]
         //
         verifyException("com.google.common.base.Preconditions", e);
      }
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      PureFunctionIdentifier pureFunctionIdentifier0 = new PureFunctionIdentifier(compiler0, (DefinitionProvider) null);
      Node node0 = compiler0.parseTestCode("function JSCompiler_identityFn() {  return function(JSCompiler_identityFn_value) {return JSCompiler_identityFn_value}}");
      Node node1 = new Node(43);
      pureFunctionIdentifier0.process(node1, node0);
      String string0 = pureFunctionIdentifier0.getDebugReport();
      assertNotNull(string0);
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      PureFunctionIdentifier pureFunctionIdentifier0 = new PureFunctionIdentifier(compiler0, (DefinitionProvider) null);
      Node node0 = compiler0.parseTestCode("2a-@5K|.R9C)KF");
      pureFunctionIdentifier0.process(node0, node0);
      // Undeclared exception!
      try { 
        pureFunctionIdentifier0.process(node0, node0);
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // It is illegal to call PureFunctionIdentifier.process twice the same instance.  Please use a new PureFunctionIdentifier instance each time.
         //
         verifyException("com.google.javascript.jscomp.PureFunctionIdentifier", e);
      }
  }

  @Test(timeout = 4000)
  public void test3()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      PureFunctionIdentifier pureFunctionIdentifier0 = new PureFunctionIdentifier(compiler0, (DefinitionProvider) null);
      Node node0 = compiler0.parseTestCode("function JSCompiler_identityFn() {  return function(JSCompiler_identityFn_value) {return JSCompiler_identityFn_value}}");
      Node node1 = new Node(43);
      pureFunctionIdentifier0.process(node0, node1);
      String string0 = pureFunctionIdentifier0.getDebugReport();
      assertNotNull(string0);
  }

  @Test(timeout = 4000)
  public void test4()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      PureFunctionIdentifier pureFunctionIdentifier0 = new PureFunctionIdentifier(compiler0, (DefinitionProvider) null);
      Node node0 = Node.newNumber((double) 30);
      Node node1 = new Node(30, node0, node0, node0);
      pureFunctionIdentifier0.process(node0, node1);
      assertEquals(21, Node.LOCALCOUNT_PROP);
  }

  @Test(timeout = 4000)
  public void test5()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      PureFunctionIdentifier pureFunctionIdentifier0 = new PureFunctionIdentifier(compiler0, (DefinitionProvider) null);
      Node node0 = new Node(37);
      // Undeclared exception!
      try { 
        pureFunctionIdentifier0.process(node0, node0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.NodeUtil", e);
      }
  }

  @Test(timeout = 4000)
  public void test6()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      PureFunctionIdentifier pureFunctionIdentifier0 = new PureFunctionIdentifier(compiler0, (DefinitionProvider) null);
      Node node0 = new Node(102);
      pureFunctionIdentifier0.process(node0, node0);
      assertFalse(node0.isNoSideEffectsCall());
  }
}
