/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 04:41:33 GMT 2023
 */

package com.google.javascript.jscomp;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.google.javascript.jscomp.AbstractCompiler;
import com.google.javascript.jscomp.CollapseVariableDeclarations;
import com.google.javascript.jscomp.Compiler;
import com.google.javascript.jscomp.Normalize;
import com.google.javascript.rhino.Node;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class CollapseVariableDeclarations_ESTest extends CollapseVariableDeclarations_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = new Node(118);
      Node node1 = new Node(118);
      Node node2 = new Node(118, node1, node0, 39, 1);
      CollapseVariableDeclarations collapseVariableDeclarations0 = new CollapseVariableDeclarations(compiler0);
      // Undeclared exception!
      try { 
        collapseVariableDeclarations0.process(node2, node2);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
      }
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      AbstractCompiler.LifeCycleStage abstractCompiler_LifeCycleStage0 = AbstractCompiler.LifeCycleStage.NORMALIZED_OBFUSCATED;
      compiler0.setLifeCycleStage(abstractCompiler_LifeCycleStage0);
      CollapseVariableDeclarations collapseVariableDeclarations0 = null;
      try {
        collapseVariableDeclarations0 = new CollapseVariableDeclarations(compiler0);
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.common.base.Preconditions", e);
      }
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      CollapseVariableDeclarations collapseVariableDeclarations0 = new CollapseVariableDeclarations(compiler0);
      Node node0 = Node.newString(118, "ormgU^3pE>c1@");
      Node node1 = Normalize.parseAndNormalizeTestCode(compiler0, "gJ74rj=lI3", "ormgU^3pE>c1@");
      Node node2 = new Node((-4069), node0, node1, (-1212), 39);
      collapseVariableDeclarations0.process(node0, node2);
  }
}