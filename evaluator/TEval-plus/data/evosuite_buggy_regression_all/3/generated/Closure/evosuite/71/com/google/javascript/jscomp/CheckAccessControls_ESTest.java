/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 17:13:34 GMT 2023
 */

package com.google.javascript.jscomp;

import org.junit.Test;
import static org.junit.Assert.*;
import com.google.javascript.jscomp.CheckAccessControls;
import com.google.javascript.jscomp.Compiler;
import com.google.javascript.rhino.Node;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class CheckAccessControls_ESTest extends CheckAccessControls_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseSyntheticCode("com.google.javas=ribt.jscomp.SpecAalize;odule", "com.google.javas=ribt.jscomp.SpecAalize;odule");
      CheckAccessControls checkAccessControls0 = new CheckAccessControls(compiler0);
      checkAccessControls0.process(node0, node0);
      assertEquals(2, Node.SPECIALCALL_WITH);
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      compiler0.getErrorManager();
      Node node0 = new Node(30);
      CheckAccessControls checkAccessControls0 = new CheckAccessControls(compiler0);
      checkAccessControls0.hotSwapScript(node0);
      assertEquals(0, Node.SIDE_EFFECTS_ALL);
  }
}
