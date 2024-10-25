/*
 * This file was automatically generated by EvoSuite
 * Wed Sep 27 00:05:39 GMT 2023
 */

package com.google.javascript.jscomp;

import org.junit.Test;
import static org.junit.Assert.*;
import com.google.javascript.jscomp.Compiler;
import com.google.javascript.jscomp.RemoveConstantExpressions;
import com.google.javascript.rhino.Node;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class RemoveConstantExpressions_ESTest extends RemoveConstantExpressions_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseSyntheticCode("com.google.javascript.jscomp.FunctionRewriter$GetterReducer", "com.google.javascript.jscomp.FunctionRewriter$GetterReducer");
      RemoveConstantExpressions removeConstantExpressions0 = new RemoveConstantExpressions(compiler0);
      removeConstantExpressions0.process(node0, node0);
      assertEquals(47, Node.IS_DISPATCHER);
  }
}
