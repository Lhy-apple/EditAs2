/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 13:52:10 GMT 2023
 */

package org.apache.commons.jxpath.ri.compiler;

import org.junit.Test;
import static org.junit.Assert.*;
import org.apache.commons.jxpath.ri.EvalContext;
import org.apache.commons.jxpath.ri.QName;
import org.apache.commons.jxpath.ri.axes.PrecedingOrFollowingContext;
import org.apache.commons.jxpath.ri.compiler.Constant;
import org.apache.commons.jxpath.ri.compiler.CoreFunction;
import org.apache.commons.jxpath.ri.compiler.CoreOperationGreaterThan;
import org.apache.commons.jxpath.ri.compiler.CoreOperationSubtract;
import org.apache.commons.jxpath.ri.compiler.Expression;
import org.apache.commons.jxpath.ri.compiler.NodeTypeTest;
import org.apache.commons.jxpath.ri.compiler.VariableReference;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class CoreOperationGreaterThan_ESTest extends CoreOperationGreaterThan_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      Expression[] expressionArray0 = new Expression[0];
      CoreFunction coreFunction0 = new CoreFunction((-1), expressionArray0);
      Float float0 = new Float((-257.0));
      Constant constant0 = new Constant(float0);
      CoreOperationSubtract coreOperationSubtract0 = new CoreOperationSubtract(coreFunction0, constant0);
      CoreOperationGreaterThan coreOperationGreaterThan0 = new CoreOperationGreaterThan(coreOperationSubtract0, constant0);
      NodeTypeTest nodeTypeTest0 = new NodeTypeTest((-3092));
      PrecedingOrFollowingContext precedingOrFollowingContext0 = new PrecedingOrFollowingContext((EvalContext) null, nodeTypeTest0, true);
      Object object0 = coreOperationGreaterThan0.computeValue(precedingOrFollowingContext0);
      assertEquals(true, object0);
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      VariableReference variableReference0 = new VariableReference((QName) null);
      CoreOperationGreaterThan coreOperationGreaterThan0 = new CoreOperationGreaterThan(variableReference0, variableReference0);
      coreOperationGreaterThan0.toString();
      assertEquals(">", coreOperationGreaterThan0.getSymbol());
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      Double double0 = Expression.NOT_A_NUMBER;
      Constant constant0 = new Constant(double0);
      CoreOperationGreaterThan coreOperationGreaterThan0 = new CoreOperationGreaterThan(constant0, constant0);
      Object object0 = coreOperationGreaterThan0.compute((EvalContext) null);
      assertEquals(false, object0);
  }
}