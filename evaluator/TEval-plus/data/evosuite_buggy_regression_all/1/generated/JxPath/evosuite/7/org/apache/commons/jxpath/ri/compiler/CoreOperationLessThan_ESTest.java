/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 13:52:22 GMT 2023
 */

package org.apache.commons.jxpath.ri.compiler;

import org.junit.Test;
import static org.junit.Assert.*;
import org.apache.commons.jxpath.ri.EvalContext;
import org.apache.commons.jxpath.ri.compiler.Constant;
import org.apache.commons.jxpath.ri.compiler.CoreOperationLessThan;
import org.apache.commons.jxpath.ri.compiler.Expression;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class CoreOperationLessThan_ESTest extends CoreOperationLessThan_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      Double double0 = Expression.ZERO;
      Constant constant0 = new Constant(double0);
      CoreOperationLessThan coreOperationLessThan0 = new CoreOperationLessThan(constant0, constant0);
      String string0 = coreOperationLessThan0.getSymbol();
      assertEquals("<", string0);
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      Double double0 = Expression.ZERO;
      Constant constant0 = new Constant(double0);
      CoreOperationLessThan coreOperationLessThan0 = new CoreOperationLessThan(constant0, constant0);
      CoreOperationLessThan coreOperationLessThan1 = new CoreOperationLessThan(constant0, coreOperationLessThan0);
      Object object0 = coreOperationLessThan1.computeValue((EvalContext) null);
      assertEquals(true, object0);
  }
}
