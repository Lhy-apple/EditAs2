/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 15:15:31 GMT 2023
 */

package org.apache.commons.jxpath.ri.compiler;

import org.junit.Test;
import static org.junit.Assert.*;
import org.apache.commons.jxpath.ri.compiler.Constant;
import org.apache.commons.jxpath.ri.compiler.CoreOperationGreaterThan;
import org.apache.commons.jxpath.ri.compiler.CoreOperationLessThanOrEqual;
import org.apache.commons.jxpath.ri.compiler.Expression;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class CoreOperationRelationalExpression_ESTest extends CoreOperationRelationalExpression_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      Constant constant0 = new Constant("S77gZ$N%C");
      CoreOperationLessThanOrEqual coreOperationLessThanOrEqual0 = new CoreOperationLessThanOrEqual((Expression) null, constant0);
      int int0 = coreOperationLessThanOrEqual0.getPrecedence();
      assertEquals(3, int0);
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      Double double0 = Expression.ONE;
      Constant constant0 = new Constant(double0);
      CoreOperationGreaterThan coreOperationGreaterThan0 = new CoreOperationGreaterThan(constant0, constant0);
      boolean boolean0 = coreOperationGreaterThan0.isSymmetric();
      assertFalse(boolean0);
  }
}
