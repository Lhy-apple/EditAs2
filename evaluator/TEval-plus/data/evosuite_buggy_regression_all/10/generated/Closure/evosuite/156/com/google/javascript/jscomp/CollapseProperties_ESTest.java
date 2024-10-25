/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 12:58:42 GMT 2023
 */

package com.google.javascript.jscomp;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.google.javascript.jscomp.CollapseProperties;
import com.google.javascript.jscomp.Compiler;
import com.google.javascript.rhino.Node;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class CollapseProperties_ESTest extends CollapseProperties_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      CollapseProperties collapseProperties0 = new CollapseProperties(compiler0, true, true);
      Node node0 = compiler0.parseSyntheticCode("com.google.javascript.jscomp.CollapseProperties$1", "com.google.javascript.jscomp.CollapseProperties$1");
      collapseProperties0.process(node0, node0);
      assertEquals(4, Node.DESCENDANTS_FLAG);
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      CollapseProperties collapseProperties0 = new CollapseProperties(compiler0, false, false);
      // Undeclared exception!
      try { 
        collapseProperties0.process((Node) null, (Node) null);
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
}
