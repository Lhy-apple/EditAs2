/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 15:17:50 GMT 2023
 */

package org.apache.commons.jxpath.ri.model.beans;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.util.Locale;
import org.apache.commons.jxpath.JXPathBasicBeanInfo;
import org.apache.commons.jxpath.JXPathBeanInfo;
import org.apache.commons.jxpath.JXPathContext;
import org.apache.commons.jxpath.ri.QName;
import org.apache.commons.jxpath.ri.model.NodePointer;
import org.apache.commons.jxpath.ri.model.VariablePointer;
import org.apache.commons.jxpath.ri.model.beans.BeanPropertyPointer;
import org.apache.commons.jxpath.ri.model.beans.NullPropertyPointer;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class PropertyPointer_ESTest extends PropertyPointer_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      Class<VariablePointer> class0 = VariablePointer.class;
      JXPathBasicBeanInfo jXPathBasicBeanInfo0 = new JXPathBasicBeanInfo(class0);
      BeanPropertyPointer beanPropertyPointer0 = new BeanPropertyPointer((NodePointer) null, jXPathBasicBeanInfo0);
      beanPropertyPointer0.setPropertyIndex(5);
      // Undeclared exception!
      try { 
        beanPropertyPointer0.getLength();
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.apache.commons.jxpath.ri.model.beans.PropertyPointer", e);
      }
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      NullPropertyPointer nullPropertyPointer0 = new NullPropertyPointer((NodePointer) null);
      // Undeclared exception!
      try { 
        nullPropertyPointer0.compareChildNodePointers((NodePointer) null, (NodePointer) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.apache.commons.jxpath.ri.model.beans.PropertyOwnerPointer", e);
      }
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      Class<String> class0 = String.class;
      JXPathBasicBeanInfo jXPathBasicBeanInfo0 = new JXPathBasicBeanInfo(class0);
      BeanPropertyPointer beanPropertyPointer0 = new BeanPropertyPointer((NodePointer) null, jXPathBasicBeanInfo0);
      QName qName0 = beanPropertyPointer0.getName();
      NullPropertyPointer nullPropertyPointer0 = new NullPropertyPointer((NodePointer) null);
      // Undeclared exception!
      try { 
        beanPropertyPointer0.createChild((JXPathContext) null, qName0, 38, (Object) nullPropertyPointer0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.apache.commons.jxpath.ri.model.NodePointer", e);
      }
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      BeanPropertyPointer beanPropertyPointer0 = new BeanPropertyPointer((NodePointer) null, (JXPathBeanInfo) null);
      // Undeclared exception!
      try { 
        beanPropertyPointer0.isLeaf();
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.apache.commons.jxpath.ri.model.beans.BeanPropertyPointer", e);
      }
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      BeanPropertyPointer beanPropertyPointer0 = new BeanPropertyPointer((NodePointer) null, (JXPathBeanInfo) null);
      BeanPropertyPointer beanPropertyPointer1 = new BeanPropertyPointer(beanPropertyPointer0, (JXPathBeanInfo) null);
      beanPropertyPointer0.compareTo(beanPropertyPointer1);
      assertEquals(Integer.MIN_VALUE, beanPropertyPointer1.getPropertyIndex());
      assertEquals(Integer.MIN_VALUE, beanPropertyPointer0.getPropertyIndex());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      NullPropertyPointer nullPropertyPointer0 = new NullPropertyPointer((NodePointer) null);
      nullPropertyPointer0.bean = (Object) "<<unknown namespace>>";
      nullPropertyPointer0.getBean();
      assertEquals(Integer.MIN_VALUE, nullPropertyPointer0.getPropertyIndex());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Class<VariablePointer> class0 = VariablePointer.class;
      JXPathBasicBeanInfo jXPathBasicBeanInfo0 = new JXPathBasicBeanInfo(class0);
      BeanPropertyPointer beanPropertyPointer0 = new BeanPropertyPointer((NodePointer) null, jXPathBasicBeanInfo0);
      beanPropertyPointer0.setPropertyIndex(4);
      boolean boolean0 = beanPropertyPointer0.isActual();
      assertEquals(4, beanPropertyPointer0.getPropertyIndex());
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Class<Locale.LanguageRange> class0 = Locale.LanguageRange.class;
      JXPathBasicBeanInfo jXPathBasicBeanInfo0 = new JXPathBasicBeanInfo(class0, class0);
      BeanPropertyPointer beanPropertyPointer0 = new BeanPropertyPointer((NodePointer) null, jXPathBasicBeanInfo0);
      boolean boolean0 = beanPropertyPointer0.isActual();
      assertEquals(Integer.MIN_VALUE, beanPropertyPointer0.getPropertyIndex());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Class<String> class0 = String.class;
      JXPathBasicBeanInfo jXPathBasicBeanInfo0 = new JXPathBasicBeanInfo(class0);
      BeanPropertyPointer beanPropertyPointer0 = new BeanPropertyPointer((NodePointer) null, jXPathBasicBeanInfo0);
      // Undeclared exception!
      try { 
        beanPropertyPointer0.createPath((JXPathContext) null, (Object) null);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // Cannot set property: /* - no such property
         //
         verifyException("org.apache.commons.jxpath.ri.model.beans.BeanPropertyPointer", e);
      }
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Class<String> class0 = String.class;
      JXPathBasicBeanInfo jXPathBasicBeanInfo0 = new JXPathBasicBeanInfo(class0);
      BeanPropertyPointer beanPropertyPointer0 = new BeanPropertyPointer((NodePointer) null, jXPathBasicBeanInfo0);
      VariablePointer variablePointer0 = new VariablePointer((QName) null);
      // Undeclared exception!
      try { 
        beanPropertyPointer0.createChild((JXPathContext) null, (QName) null, (-593), (Object) variablePointer0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // Cannot set property: /* - no such property
         //
         verifyException("org.apache.commons.jxpath.ri.model.beans.BeanPropertyPointer", e);
      }
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      Class<String> class0 = String.class;
      JXPathBasicBeanInfo jXPathBasicBeanInfo0 = new JXPathBasicBeanInfo(class0);
      BeanPropertyPointer beanPropertyPointer0 = new BeanPropertyPointer((NodePointer) null, jXPathBasicBeanInfo0);
      // Undeclared exception!
      try { 
        beanPropertyPointer0.createChild((JXPathContext) null, (QName) null, Integer.MIN_VALUE);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.apache.commons.jxpath.ri.model.NodePointer", e);
      }
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      BeanPropertyPointer beanPropertyPointer0 = new BeanPropertyPointer((NodePointer) null, (JXPathBeanInfo) null);
      QName qName0 = new QName("<<unknown namespace>>");
      // Undeclared exception!
      try { 
        beanPropertyPointer0.createChild((JXPathContext) null, qName0, 0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.apache.commons.jxpath.ri.model.beans.BeanPropertyPointer", e);
      }
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      NullPropertyPointer nullPropertyPointer0 = new NullPropertyPointer((NodePointer) null);
      boolean boolean0 = nullPropertyPointer0.equals(nullPropertyPointer0);
      assertEquals(Integer.MIN_VALUE, nullPropertyPointer0.getPropertyIndex());
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      NullPropertyPointer nullPropertyPointer0 = new NullPropertyPointer((NodePointer) null);
      NullPropertyPointer nullPropertyPointer1 = new NullPropertyPointer(nullPropertyPointer0);
      boolean boolean0 = nullPropertyPointer1.equals(nullPropertyPointer0);
      assertFalse(boolean0);
      assertEquals(Integer.MIN_VALUE, nullPropertyPointer1.getPropertyIndex());
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      NullPropertyPointer nullPropertyPointer0 = new NullPropertyPointer((NodePointer) null);
      NullPropertyPointer nullPropertyPointer1 = new NullPropertyPointer(nullPropertyPointer0);
      boolean boolean0 = nullPropertyPointer0.equals(nullPropertyPointer1);
      assertEquals(Integer.MIN_VALUE, nullPropertyPointer1.getPropertyIndex());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      NullPropertyPointer nullPropertyPointer0 = new NullPropertyPointer((NodePointer) null);
      NullPropertyPointer nullPropertyPointer1 = new NullPropertyPointer((NodePointer) null);
      NullPropertyPointer nullPropertyPointer2 = new NullPropertyPointer(nullPropertyPointer0);
      NullPropertyPointer nullPropertyPointer3 = new NullPropertyPointer(nullPropertyPointer1);
      boolean boolean0 = nullPropertyPointer2.equals(nullPropertyPointer3);
      assertTrue(boolean0);
      assertEquals(Integer.MIN_VALUE, nullPropertyPointer3.getPropertyIndex());
      assertFalse(nullPropertyPointer3.equals((Object)nullPropertyPointer1));
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      NullPropertyPointer nullPropertyPointer0 = new NullPropertyPointer((NodePointer) null);
      nullPropertyPointer0.propertyIndex = 2412;
      NullPropertyPointer nullPropertyPointer1 = new NullPropertyPointer((NodePointer) null);
      boolean boolean0 = nullPropertyPointer0.equals(nullPropertyPointer1);
      assertEquals(2412, nullPropertyPointer0.getPropertyIndex());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      NullPropertyPointer nullPropertyPointer0 = new NullPropertyPointer((NodePointer) null);
      nullPropertyPointer0.setPropertyName("<<unknown namespace>>");
      NullPropertyPointer nullPropertyPointer1 = new NullPropertyPointer((NodePointer) null);
      boolean boolean0 = nullPropertyPointer1.equals(nullPropertyPointer0);
      assertEquals(Integer.MIN_VALUE, nullPropertyPointer1.getPropertyIndex());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      Class<Object> class0 = Object.class;
      JXPathBasicBeanInfo jXPathBasicBeanInfo0 = new JXPathBasicBeanInfo(class0, true);
      BeanPropertyPointer beanPropertyPointer0 = new BeanPropertyPointer((NodePointer) null, jXPathBasicBeanInfo0);
      NullPropertyPointer nullPropertyPointer0 = new NullPropertyPointer((NodePointer) null);
      nullPropertyPointer0.setIndex((-1));
      boolean boolean0 = nullPropertyPointer0.equals(beanPropertyPointer0);
      assertFalse(boolean0);
      assertEquals(Integer.MIN_VALUE, nullPropertyPointer0.getPropertyIndex());
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      NullPropertyPointer nullPropertyPointer0 = new NullPropertyPointer((NodePointer) null);
      nullPropertyPointer0.setIndex(36);
      NullPropertyPointer nullPropertyPointer1 = new NullPropertyPointer((NodePointer) null);
      boolean boolean0 = nullPropertyPointer1.equals(nullPropertyPointer0);
      assertFalse(nullPropertyPointer0.equals((Object)nullPropertyPointer1));
      assertFalse(boolean0);
      assertEquals(Integer.MIN_VALUE, nullPropertyPointer1.getPropertyIndex());
  }
}