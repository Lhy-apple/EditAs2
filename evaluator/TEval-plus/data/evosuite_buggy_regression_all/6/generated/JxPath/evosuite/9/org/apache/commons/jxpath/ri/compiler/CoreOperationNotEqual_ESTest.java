/*
 * This file was automatically generated by EvoSuite
 * Wed Sep 27 00:39:25 GMT 2023
 */

package org.apache.commons.jxpath.ri.compiler;

import org.junit.Test;
import static org.junit.Assert.*;
import org.apache.commons.jxpath.ri.EvalContext;
import org.apache.commons.jxpath.ri.QName;
import org.apache.commons.jxpath.ri.axes.PrecedingOrFollowingContext;
import org.apache.commons.jxpath.ri.axes.SelfContext;
import org.apache.commons.jxpath.ri.compiler.Constant;
import org.apache.commons.jxpath.ri.compiler.CoreOperationNotEqual;
import org.apache.commons.jxpath.ri.compiler.NodeNameTest;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class CoreOperationNotEqual_ESTest extends CoreOperationNotEqual_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      Integer integer0 = new Integer(0);
      Constant constant0 = new Constant(integer0);
      CoreOperationNotEqual coreOperationNotEqual0 = new CoreOperationNotEqual(constant0, constant0);
      String string0 = coreOperationNotEqual0.getSymbol();
      assertEquals("!=", string0);
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      Integer integer0 = new Integer(0);
      Constant constant0 = new Constant(integer0);
      Constant constant1 = new Constant("&apos;");
      CoreOperationNotEqual coreOperationNotEqual0 = new CoreOperationNotEqual(constant0, constant1);
      Object object0 = coreOperationNotEqual0.computeValue((EvalContext) null);
      assertEquals(true, object0);
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      Constant constant0 = new Constant("org.apache.commons.jxpath.ri.compiler.CoreOperationNotEqual");
      CoreOperationNotEqual coreOperationNotEqual0 = new CoreOperationNotEqual(constant0, constant0);
      QName qName0 = new QName("org.apache.commons.jxpath.ri.compiler.CoreOperationNotEqual", "!=");
      NodeNameTest nodeNameTest0 = new NodeNameTest(qName0);
      SelfContext selfContext0 = new SelfContext((EvalContext) null, nodeNameTest0);
      PrecedingOrFollowingContext precedingOrFollowingContext0 = new PrecedingOrFollowingContext(selfContext0, nodeNameTest0, true);
      Object object0 = coreOperationNotEqual0.computeValue(precedingOrFollowingContext0);
      assertEquals(false, object0);
  }
}