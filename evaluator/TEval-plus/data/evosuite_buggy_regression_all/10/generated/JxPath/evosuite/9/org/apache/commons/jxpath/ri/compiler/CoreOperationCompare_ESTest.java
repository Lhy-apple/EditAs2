/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 14:02:19 GMT 2023
 */

package org.apache.commons.jxpath.ri.compiler;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.util.Locale;
import org.apache.commons.jxpath.JXPathContext;
import org.apache.commons.jxpath.ri.EvalContext;
import org.apache.commons.jxpath.ri.JXPathContextReferenceImpl;
import org.apache.commons.jxpath.ri.QName;
import org.apache.commons.jxpath.ri.axes.InitialContext;
import org.apache.commons.jxpath.ri.axes.RootContext;
import org.apache.commons.jxpath.ri.compiler.Constant;
import org.apache.commons.jxpath.ri.compiler.CoreOperationEqual;
import org.apache.commons.jxpath.ri.compiler.CoreOperationSubtract;
import org.apache.commons.jxpath.ri.compiler.Expression;
import org.apache.commons.jxpath.ri.compiler.NameAttributeTest;
import org.apache.commons.jxpath.ri.compiler.VariableReference;
import org.apache.commons.jxpath.ri.model.NodePointer;
import org.apache.commons.jxpath.ri.model.VariablePointer;
import org.apache.commons.jxpath.ri.model.beans.BeanPointer;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class CoreOperationCompare_ESTest extends CoreOperationCompare_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      QName qName0 = new QName("KHb'}qtm");
      VariableReference variableReference0 = new VariableReference(qName0);
      NameAttributeTest nameAttributeTest0 = new NameAttributeTest(variableReference0, variableReference0);
      CoreOperationEqual coreOperationEqual0 = new CoreOperationEqual(nameAttributeTest0, variableReference0);
      JXPathContextReferenceImpl jXPathContextReferenceImpl0 = (JXPathContextReferenceImpl)JXPathContext.newContext((Object) coreOperationEqual0);
      Locale locale0 = Locale.CANADA_FRENCH;
      BeanPointer beanPointer0 = (BeanPointer)NodePointer.newNodePointer(qName0, qName0, locale0);
      RootContext rootContext0 = new RootContext(jXPathContextReferenceImpl0, beanPointer0);
      InitialContext initialContext0 = (InitialContext)rootContext0.getConstantContext("KHb'}qtm");
      Boolean boolean0 = (Boolean)coreOperationEqual0.computeValue(initialContext0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      NameAttributeTest nameAttributeTest0 = new NameAttributeTest((Expression) null, (Expression) null);
      boolean boolean0 = nameAttributeTest0.isSymmetric();
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      Double double0 = Expression.NOT_A_NUMBER;
      Constant constant0 = new Constant(double0);
      NameAttributeTest nameAttributeTest0 = new NameAttributeTest(constant0, constant0);
      int int0 = nameAttributeTest0.getPrecedence();
      assertEquals(2, int0);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      QName qName0 = new QName("KHb'}qtm");
      VariableReference variableReference0 = new VariableReference(qName0);
      NameAttributeTest nameAttributeTest0 = new NameAttributeTest(variableReference0, variableReference0);
      JXPathContextReferenceImpl jXPathContextReferenceImpl0 = (JXPathContextReferenceImpl)JXPathContext.newContext((Object) nameAttributeTest0);
      Locale locale0 = Locale.CANADA_FRENCH;
      BeanPointer beanPointer0 = (BeanPointer)NodePointer.newNodePointer(qName0, qName0, locale0);
      RootContext rootContext0 = new RootContext(jXPathContextReferenceImpl0, beanPointer0);
      boolean boolean0 = nameAttributeTest0.equal(rootContext0, variableReference0, nameAttributeTest0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      Double double0 = Expression.ONE;
      Constant constant0 = new Constant(double0);
      CoreOperationEqual coreOperationEqual0 = new CoreOperationEqual(constant0, constant0);
      boolean boolean0 = coreOperationEqual0.equal((EvalContext) null, constant0, coreOperationEqual0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      Constant constant0 = new Constant("");
      CoreOperationSubtract coreOperationSubtract0 = new CoreOperationSubtract(constant0, constant0);
      NameAttributeTest nameAttributeTest0 = new NameAttributeTest(coreOperationSubtract0, constant0);
      JXPathContextReferenceImpl jXPathContextReferenceImpl0 = (JXPathContextReferenceImpl)JXPathContext.newContext((Object) "");
      QName qName0 = new QName("", "");
      Locale locale0 = Locale.ROOT;
      BeanPointer beanPointer0 = (BeanPointer)NodePointer.newNodePointer(qName0, constant0, locale0);
      RootContext rootContext0 = new RootContext(jXPathContextReferenceImpl0, beanPointer0);
      InitialContext initialContext0 = (InitialContext)rootContext0.getAbsoluteRootContext();
      InitialContext initialContext1 = new InitialContext(initialContext0);
      boolean boolean0 = nameAttributeTest0.findMatch(initialContext0, initialContext1);
      assertEquals(1, initialContext1.getPosition());
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Constant constant0 = new Constant("");
      CoreOperationSubtract coreOperationSubtract0 = new CoreOperationSubtract(constant0, constant0);
      NameAttributeTest nameAttributeTest0 = new NameAttributeTest(coreOperationSubtract0, constant0);
      JXPathContextReferenceImpl jXPathContextReferenceImpl0 = (JXPathContextReferenceImpl)JXPathContext.newContext((Object) "");
      QName qName0 = new QName("", "");
      Locale locale0 = Locale.ROOT;
      BeanPointer beanPointer0 = (BeanPointer)NodePointer.newNodePointer(qName0, constant0, locale0);
      RootContext rootContext0 = new RootContext(jXPathContextReferenceImpl0, beanPointer0);
      InitialContext initialContext0 = (InitialContext)rootContext0.getAbsoluteRootContext();
      InitialContext initialContext1 = new InitialContext(rootContext0);
      boolean boolean0 = nameAttributeTest0.findMatch(initialContext1, initialContext0);
      assertEquals(2, initialContext0.getPosition());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Double double0 = Expression.NOT_A_NUMBER;
      Constant constant0 = new Constant(double0);
      CoreOperationEqual coreOperationEqual0 = new CoreOperationEqual(constant0, constant0);
      VariablePointer variablePointer0 = new VariablePointer((QName) null);
      // Undeclared exception!
      try { 
        coreOperationEqual0.equal(variablePointer0, double0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // Undefined variable: null
         //
         verifyException("org.apache.commons.jxpath.ri.model.VariablePointer$1", e);
      }
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Double double0 = Expression.ZERO;
      Constant constant0 = new Constant(double0);
      CoreOperationEqual coreOperationEqual0 = new CoreOperationEqual(constant0, constant0);
      NameAttributeTest nameAttributeTest0 = new NameAttributeTest(constant0, constant0);
      boolean boolean0 = nameAttributeTest0.equal((EvalContext) null, coreOperationEqual0, constant0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      QName qName0 = new QName("KHb'}qtm");
      VariableReference variableReference0 = new VariableReference(qName0);
      NameAttributeTest nameAttributeTest0 = new NameAttributeTest(variableReference0, variableReference0);
      Short short0 = new Short((short)1);
      boolean boolean0 = nameAttributeTest0.equal(short0, qName0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      Double double0 = Expression.ZERO;
      Constant constant0 = new Constant(double0);
      CoreOperationEqual coreOperationEqual0 = new CoreOperationEqual(constant0, constant0);
      boolean boolean0 = coreOperationEqual0.equal(constant0, double0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      Double double0 = Expression.ZERO;
      Constant constant0 = new Constant(double0);
      CoreOperationEqual coreOperationEqual0 = new CoreOperationEqual(constant0, constant0);
      boolean boolean0 = coreOperationEqual0.equal("=", constant0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      Double double0 = Expression.NOT_A_NUMBER;
      Constant constant0 = new Constant(double0);
      CoreOperationEqual coreOperationEqual0 = new CoreOperationEqual(constant0, constant0);
      boolean boolean0 = coreOperationEqual0.equal(constant0, coreOperationEqual0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      Double double0 = Expression.NOT_A_NUMBER;
      Constant constant0 = new Constant(double0);
      CoreOperationEqual coreOperationEqual0 = new CoreOperationEqual(constant0, constant0);
      boolean boolean0 = coreOperationEqual0.equal((Object) null, constant0);
      assertFalse(boolean0);
  }
}
