/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 04:42:20 GMT 2023
 */

package com.google.javascript.jscomp;

import org.junit.Test;
import static org.junit.Assert.*;
import com.google.javascript.jscomp.AbstractCompiler;
import com.google.javascript.jscomp.AbstractPeepholeOptimization;
import com.google.javascript.jscomp.Compiler;
import com.google.javascript.jscomp.Normalize;
import com.google.javascript.jscomp.PeepholeOptimizationsPass;
import com.google.javascript.jscomp.PeepholeRemoveDeadCode;
import com.google.javascript.rhino.Node;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class PeepholeOptimizationsPass_ESTest extends PeepholeOptimizationsPass_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      AbstractPeepholeOptimization[] abstractPeepholeOptimizationArray0 = new AbstractPeepholeOptimization[1];
      PeepholeRemoveDeadCode peepholeRemoveDeadCode0 = new PeepholeRemoveDeadCode();
      abstractPeepholeOptimizationArray0[0] = (AbstractPeepholeOptimization) peepholeRemoveDeadCode0;
      PeepholeOptimizationsPass peepholeOptimizationsPass0 = new PeepholeOptimizationsPass(compiler0, abstractPeepholeOptimizationArray0);
      Node node0 = Normalize.parseAndNormalizeTestCode(compiler0, "com.google.javascript.jscomp.Denormalize$StripConstantAnnotations", "com.google.javascript.jscomp.Denormalize$StripConstantAnnotations");
      peepholeOptimizationsPass0.process(node0, node0);
      assertFalse(node0.hasChildren());
      assertEquals(0, node0.getChildCount());
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      AbstractPeepholeOptimization[] abstractPeepholeOptimizationArray0 = new AbstractPeepholeOptimization[0];
      PeepholeOptimizationsPass peepholeOptimizationsPass0 = new PeepholeOptimizationsPass(compiler0, abstractPeepholeOptimizationArray0);
      AbstractCompiler abstractCompiler0 = peepholeOptimizationsPass0.getCompiler();
      assertSame(abstractCompiler0, compiler0);
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      AbstractPeepholeOptimization[] abstractPeepholeOptimizationArray0 = new AbstractPeepholeOptimization[0];
      PeepholeOptimizationsPass peepholeOptimizationsPass0 = new PeepholeOptimizationsPass(compiler0, abstractPeepholeOptimizationArray0);
      Node node0 = Normalize.parseAndNormalizeTestCode(compiler0, "]A6#)|S4}c1V[+a~]", "]A6#)|S4}c1V[+a~]");
      peepholeOptimizationsPass0.process(node0, node0);
      peepholeOptimizationsPass0.process(node0, node0);
      assertFalse(node0.isExprResult());
  }
}