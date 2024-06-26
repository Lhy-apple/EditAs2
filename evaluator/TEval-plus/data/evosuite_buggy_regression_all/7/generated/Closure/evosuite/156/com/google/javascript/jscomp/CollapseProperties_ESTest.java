/*
 * This file was automatically generated by EvoSuite
 * Sat Jul 29 18:24:30 GMT 2023
 */

package com.google.javascript.jscomp;

import org.junit.Test;
import static org.junit.Assert.*;
import com.google.javascript.jscomp.CollapseProperties;
import com.google.javascript.jscomp.Compiler;
import com.google.javascript.jscomp.Normalize;
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
      Node node0 = Normalize.parseAndNormalizeTestCode(compiler0, "", "");
      collapseProperties0.process(node0, node0);
      assertEquals(49, Node.FREE_CALL);
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      CollapseProperties collapseProperties0 = new CollapseProperties(compiler0, false, false);
      Node node0 = Normalize.parseAndNormalizeTestCode(compiler0, "", "");
      collapseProperties0.process(node0, node0);
      assertEquals(0, Node.LABEL_ID_PROP);
  }
}
