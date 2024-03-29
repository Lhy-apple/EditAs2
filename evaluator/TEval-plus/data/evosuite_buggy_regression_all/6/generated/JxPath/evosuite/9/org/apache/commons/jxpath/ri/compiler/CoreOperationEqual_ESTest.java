/*
 * This file was automatically generated by EvoSuite
 * Wed Sep 27 00:39:20 GMT 2023
 */

package org.apache.commons.jxpath.ri.compiler;

import org.junit.Test;
import static org.junit.Assert.*;
import org.apache.commons.jxpath.ri.EvalContext;
import org.apache.commons.jxpath.ri.compiler.Constant;
import org.apache.commons.jxpath.ri.compiler.CoreOperationSubtract;
import org.apache.commons.jxpath.ri.compiler.NameAttributeTest;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class CoreOperationEqual_ESTest extends CoreOperationEqual_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      Constant constant0 = new Constant("t#|cJl#Wv2[>");
      NameAttributeTest nameAttributeTest0 = new NameAttributeTest(constant0, constant0);
      String string0 = nameAttributeTest0.getSymbol();
      assertEquals("=", string0);
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      Constant constant0 = new Constant("0}+.jhVuM<sheGnt");
      CoreOperationSubtract coreOperationSubtract0 = new CoreOperationSubtract(constant0, constant0);
      NameAttributeTest nameAttributeTest0 = new NameAttributeTest(coreOperationSubtract0, coreOperationSubtract0);
      Object object0 = nameAttributeTest0.computeValue((EvalContext) null);
      assertEquals(false, object0);
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      Constant constant0 = new Constant("t#|cJl#Wv2[>");
      NameAttributeTest nameAttributeTest0 = new NameAttributeTest(constant0, constant0);
      Object object0 = nameAttributeTest0.computeValue((EvalContext) null);
      assertEquals(true, object0);
  }
}
