/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 06:18:37 GMT 2023
 */

package org.apache.commons.jxpath.ri.compiler;

import org.junit.Test;
import static org.junit.Assert.*;
import org.apache.commons.jxpath.ri.EvalContext;
import org.apache.commons.jxpath.ri.axes.PrecedingOrFollowingContext;
import org.apache.commons.jxpath.ri.axes.SelfContext;
import org.apache.commons.jxpath.ri.compiler.Constant;
import org.apache.commons.jxpath.ri.compiler.CoreOperationGreaterThan;
import org.apache.commons.jxpath.ri.compiler.CoreOperationOr;
import org.apache.commons.jxpath.ri.compiler.Expression;
import org.apache.commons.jxpath.ri.compiler.NameAttributeTest;
import org.apache.commons.jxpath.ri.compiler.NodeTypeTest;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class CoreOperationEqual_ESTest extends CoreOperationEqual_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      NameAttributeTest nameAttributeTest0 = new NameAttributeTest((Expression) null, (Expression) null);
      String string0 = nameAttributeTest0.getSymbol();
      assertEquals("=", string0);
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      Expression[] expressionArray0 = new Expression[6];
      Constant constant0 = new Constant("org.apache.commons.jxpath.ri.compiler.CoreOperationEqual");
      expressionArray0[0] = (Expression) constant0;
      CoreOperationOr coreOperationOr0 = new CoreOperationOr(expressionArray0);
      expressionArray0[2] = (Expression) coreOperationOr0;
      CoreOperationGreaterThan coreOperationGreaterThan0 = new CoreOperationGreaterThan(expressionArray0[2], expressionArray0[2]);
      expressionArray0[4] = (Expression) coreOperationGreaterThan0;
      NodeTypeTest nodeTypeTest0 = new NodeTypeTest((-1));
      NameAttributeTest nameAttributeTest0 = new NameAttributeTest(coreOperationOr0, expressionArray0[4]);
      SelfContext selfContext0 = new SelfContext((EvalContext) null, nodeTypeTest0);
      Object object0 = nameAttributeTest0.computeValue(selfContext0);
      assertEquals(false, object0);
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      Double double0 = Expression.NOT_A_NUMBER;
      Constant constant0 = new Constant(double0);
      NameAttributeTest nameAttributeTest0 = new NameAttributeTest(constant0, constant0);
      NodeTypeTest nodeTypeTest0 = new NodeTypeTest(32);
      PrecedingOrFollowingContext precedingOrFollowingContext0 = new PrecedingOrFollowingContext((EvalContext) null, nodeTypeTest0, false);
      Object object0 = nameAttributeTest0.computeValue(precedingOrFollowingContext0);
      assertEquals(true, object0);
  }
}
