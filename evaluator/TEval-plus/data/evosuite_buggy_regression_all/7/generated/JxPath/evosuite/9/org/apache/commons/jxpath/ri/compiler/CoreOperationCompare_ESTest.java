/*
 * This file was automatically generated by EvoSuite
 * Sat Jul 29 19:31:56 GMT 2023
 */

package org.apache.commons.jxpath.ri.compiler;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.util.Iterator;
import org.apache.commons.jxpath.JXPathContext;
import org.apache.commons.jxpath.Pointer;
import org.apache.commons.jxpath.Variables;
import org.apache.commons.jxpath.ri.EvalContext;
import org.apache.commons.jxpath.ri.JXPathContextReferenceImpl;
import org.apache.commons.jxpath.ri.QName;
import org.apache.commons.jxpath.ri.axes.InitialContext;
import org.apache.commons.jxpath.ri.axes.PredicateContext;
import org.apache.commons.jxpath.ri.compiler.Constant;
import org.apache.commons.jxpath.ri.compiler.CoreOperationNotEqual;
import org.apache.commons.jxpath.ri.compiler.Expression;
import org.apache.commons.jxpath.ri.model.VariablePointer;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class CoreOperationCompare_ESTest extends CoreOperationCompare_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      CoreOperationNotEqual coreOperationNotEqual0 = new CoreOperationNotEqual((Expression) null, (Expression) null);
      CoreOperationNotEqual coreOperationNotEqual1 = new CoreOperationNotEqual(coreOperationNotEqual0, (Expression) null);
      // Undeclared exception!
      try { 
        coreOperationNotEqual1.toString();
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.evosuite.runtime.System", e);
      }
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      Constant constant0 = new Constant("o-w$kV[ECy+?{");
      JXPathContext jXPathContext0 = JXPathContext.newContext((JXPathContext) null, (Object) null);
      JXPathContextReferenceImpl jXPathContextReferenceImpl0 = new JXPathContextReferenceImpl(jXPathContext0, "o-w$kV[ECy+?{", (Pointer) null);
      EvalContext evalContext0 = jXPathContextReferenceImpl0.getAbsoluteRootContext();
      CoreOperationNotEqual coreOperationNotEqual0 = new CoreOperationNotEqual(constant0, constant0);
      boolean boolean0 = coreOperationNotEqual0.contains(evalContext0, evalContext0);
      assertEquals(1, evalContext0.getPosition());
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      Constant constant0 = new Constant("o-w$kV[ECy+?{");
      JXPathContextReferenceImpl jXPathContextReferenceImpl0 = (JXPathContextReferenceImpl)JXPathContext.newContext((JXPathContext) null, (Object) null);
      InitialContext initialContext0 = (InitialContext)jXPathContextReferenceImpl0.getAbsoluteRootContext();
      Iterator iterator0 = constant0.iteratePointers(initialContext0);
      CoreOperationNotEqual coreOperationNotEqual0 = new CoreOperationNotEqual(constant0, constant0);
      boolean boolean0 = coreOperationNotEqual0.contains(iterator0, iterator0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      Constant constant0 = new Constant((String) null);
      Iterator iterator0 = constant0.iteratePointers((EvalContext) null);
      CoreOperationNotEqual coreOperationNotEqual0 = new CoreOperationNotEqual(constant0, constant0);
      boolean boolean0 = coreOperationNotEqual0.findMatch(iterator0, iterator0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      Constant constant0 = new Constant((String) null);
      PredicateContext predicateContext0 = new PredicateContext((EvalContext) null, constant0);
      CoreOperationNotEqual coreOperationNotEqual0 = new CoreOperationNotEqual(constant0, constant0);
      JXPathContextReferenceImpl jXPathContextReferenceImpl0 = (JXPathContextReferenceImpl)JXPathContext.newContext((Object) predicateContext0);
      InitialContext initialContext0 = (InitialContext)jXPathContextReferenceImpl0.getAbsoluteRootContext();
      // Undeclared exception!
      try { 
        coreOperationNotEqual0.findMatch(initialContext0, (Iterator) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.apache.commons.jxpath.ri.compiler.CoreOperationCompare", e);
      }
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      CoreOperationNotEqual coreOperationNotEqual0 = new CoreOperationNotEqual((Expression) null, (Expression) null);
      Float float0 = new Float((-197.33235));
      JXPathContext jXPathContext0 = JXPathContext.newContext((Object) float0);
      Variables variables0 = jXPathContext0.getVariables();
      QName qName0 = new QName((String) null, (String) null);
      VariablePointer variablePointer0 = new VariablePointer(variables0, qName0);
      boolean boolean0 = coreOperationNotEqual0.equal(variablePointer0, variablePointer0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      CoreOperationNotEqual coreOperationNotEqual0 = new CoreOperationNotEqual((Expression) null, (Expression) null);
      Float float0 = new Float((-197.33235));
      JXPathContext jXPathContext0 = JXPathContext.newContext((Object) float0);
      Variables variables0 = jXPathContext0.getVariables();
      QName qName0 = new QName((String) null, (String) null);
      VariablePointer variablePointer0 = new VariablePointer(qName0);
      VariablePointer variablePointer1 = new VariablePointer(variables0, qName0);
      // Undeclared exception!
      try { 
        coreOperationNotEqual0.equal(variablePointer0, variablePointer1);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // Undefined variable: null
         //
         verifyException("org.apache.commons.jxpath.ri.model.VariablePointer$1", e);
      }
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      CoreOperationNotEqual coreOperationNotEqual0 = new CoreOperationNotEqual((Expression) null, (Expression) null);
      QName qName0 = new QName((String) null, "");
      VariablePointer variablePointer0 = new VariablePointer(qName0);
      // Undeclared exception!
      try { 
        coreOperationNotEqual0.equal((Object) null, variablePointer0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // Undefined variable: 
         //
         verifyException("org.apache.commons.jxpath.ri.model.VariablePointer$1", e);
      }
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      CoreOperationNotEqual coreOperationNotEqual0 = new CoreOperationNotEqual((Expression) null, (Expression) null);
      Boolean boolean0 = Boolean.TRUE;
      boolean boolean1 = coreOperationNotEqual0.equal(boolean0, (Object) null);
      assertFalse(boolean1);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Constant constant0 = new Constant((String) null);
      CoreOperationNotEqual coreOperationNotEqual0 = new CoreOperationNotEqual(constant0, constant0);
      boolean boolean0 = coreOperationNotEqual0.equal((EvalContext) null, constant0, coreOperationNotEqual0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      CoreOperationNotEqual coreOperationNotEqual0 = new CoreOperationNotEqual((Expression) null, (Expression) null);
      Double double0 = Expression.ZERO;
      boolean boolean0 = coreOperationNotEqual0.equal(double0, (Object) null);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      CoreOperationNotEqual coreOperationNotEqual0 = new CoreOperationNotEqual((Expression) null, (Expression) null);
      Short short0 = new Short((short) (-3128));
      boolean boolean0 = coreOperationNotEqual0.equal((Object) null, short0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      Constant constant0 = new Constant(")5i;m|C~xJa!pXlyPK");
      CoreOperationNotEqual coreOperationNotEqual0 = new CoreOperationNotEqual(constant0, constant0);
      boolean boolean0 = coreOperationNotEqual0.equal(constant0, ")5i;m|C~xJa!pXlyPK");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      CoreOperationNotEqual coreOperationNotEqual0 = new CoreOperationNotEqual((Expression) null, (Expression) null);
      boolean boolean0 = coreOperationNotEqual0.equal((Object) null, coreOperationNotEqual0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      CoreOperationNotEqual coreOperationNotEqual0 = new CoreOperationNotEqual((Expression) null, (Expression) null);
      boolean boolean0 = coreOperationNotEqual0.equal(coreOperationNotEqual0, (Object) null);
      assertFalse(boolean0);
  }
}
