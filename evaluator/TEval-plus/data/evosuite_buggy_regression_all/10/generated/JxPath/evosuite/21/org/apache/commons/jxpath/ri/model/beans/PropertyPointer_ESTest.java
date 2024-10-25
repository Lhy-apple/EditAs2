/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 14:04:57 GMT 2023
 */

package org.apache.commons.jxpath.ri.model.beans;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
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
      Class<BeanPropertyPointer> class0 = BeanPropertyPointer.class;
      JXPathBasicBeanInfo jXPathBasicBeanInfo0 = new JXPathBasicBeanInfo(class0);
      BeanPropertyPointer beanPropertyPointer0 = new BeanPropertyPointer((NodePointer) null, jXPathBasicBeanInfo0);
      beanPropertyPointer0.setPropertyIndex(3);
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
  public void test03()  throws Throwable  {
      QName qName0 = new QName("", "");
      VariablePointer variablePointer0 = new VariablePointer(qName0);
      NullPropertyPointer nullPropertyPointer0 = new NullPropertyPointer(variablePointer0);
      nullPropertyPointer0.hashCode();
      assertEquals(Integer.MIN_VALUE, nullPropertyPointer0.getPropertyIndex());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      Class<Integer> class0 = Integer.class;
      JXPathBasicBeanInfo jXPathBasicBeanInfo0 = new JXPathBasicBeanInfo(class0);
      BeanPropertyPointer beanPropertyPointer0 = new BeanPropertyPointer((NodePointer) null, jXPathBasicBeanInfo0);
      QName qName0 = new QName("<<unknown namespace>>", "Factory ");
      VariablePointer variablePointer0 = new VariablePointer(qName0);
      beanPropertyPointer0.bean = (Object) variablePointer0;
      beanPropertyPointer0.getBean();
      assertEquals(Integer.MIN_VALUE, beanPropertyPointer0.getPropertyIndex());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      Class<NodePointer> class0 = NodePointer.class;
      JXPathBasicBeanInfo jXPathBasicBeanInfo0 = new JXPathBasicBeanInfo(class0, true);
      BeanPropertyPointer beanPropertyPointer0 = new BeanPropertyPointer((NodePointer) null, jXPathBasicBeanInfo0);
      assertEquals(Integer.MIN_VALUE, beanPropertyPointer0.getPropertyIndex());
      
      beanPropertyPointer0.setPropertyIndex(0);
      boolean boolean0 = beanPropertyPointer0.isActual();
      assertEquals(0, beanPropertyPointer0.getPropertyIndex());
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Class<String> class0 = String.class;
      JXPathBasicBeanInfo jXPathBasicBeanInfo0 = new JXPathBasicBeanInfo(class0);
      BeanPropertyPointer beanPropertyPointer0 = new BeanPropertyPointer((NodePointer) null, jXPathBasicBeanInfo0);
      boolean boolean0 = beanPropertyPointer0.isActual();
      assertEquals(Integer.MIN_VALUE, beanPropertyPointer0.getPropertyIndex());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Class<Object> class0 = Object.class;
      JXPathBasicBeanInfo jXPathBasicBeanInfo0 = new JXPathBasicBeanInfo(class0);
      BeanPropertyPointer beanPropertyPointer0 = new BeanPropertyPointer((NodePointer) null, jXPathBasicBeanInfo0);
      // Undeclared exception!
      try { 
        beanPropertyPointer0.createChild((JXPathContext) null, (QName) null, 0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.apache.commons.jxpath.ri.model.NodePointer", e);
      }
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      BeanPropertyPointer beanPropertyPointer0 = new BeanPropertyPointer((NodePointer) null, (JXPathBeanInfo) null);
      // Undeclared exception!
      try { 
        beanPropertyPointer0.createPath((JXPathContext) null, (Object) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.apache.commons.jxpath.ri.model.beans.BeanPropertyPointer", e);
      }
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Class<Object> class0 = Object.class;
      JXPathBasicBeanInfo jXPathBasicBeanInfo0 = new JXPathBasicBeanInfo(class0);
      BeanPropertyPointer beanPropertyPointer0 = new BeanPropertyPointer((NodePointer) null, jXPathBasicBeanInfo0);
      // Undeclared exception!
      try { 
        beanPropertyPointer0.createChild((JXPathContext) null, (QName) null, 3485, (Object) class0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.apache.commons.jxpath.ri.model.NodePointer", e);
      }
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      Class<Integer> class0 = Integer.class;
      JXPathBasicBeanInfo jXPathBasicBeanInfo0 = new JXPathBasicBeanInfo(class0);
      BeanPropertyPointer beanPropertyPointer0 = new BeanPropertyPointer((NodePointer) null, jXPathBasicBeanInfo0);
      QName qName0 = new QName("<<unknown namespace>>", "<<unknown namespace>>");
      NullPropertyPointer nullPropertyPointer0 = new NullPropertyPointer((NodePointer) null);
      // Undeclared exception!
      try { 
        beanPropertyPointer0.createChild((JXPathContext) null, qName0, (-286), (Object) nullPropertyPointer0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // Cannot set property: /<<unknown namespace>>:<<unknown namespace>> - no such property
         //
         verifyException("org.apache.commons.jxpath.ri.model.beans.BeanPropertyPointer", e);
      }
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      BeanPropertyPointer beanPropertyPointer0 = new BeanPropertyPointer((NodePointer) null, (JXPathBeanInfo) null);
      NullPropertyPointer nullPropertyPointer0 = new NullPropertyPointer((NodePointer) null);
      QName qName0 = nullPropertyPointer0.getName();
      // Undeclared exception!
      try { 
        beanPropertyPointer0.createChild((JXPathContext) null, qName0, Integer.MIN_VALUE);
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
      Class<Integer> class0 = Integer.class;
      JXPathBasicBeanInfo jXPathBasicBeanInfo0 = new JXPathBasicBeanInfo(class0);
      BeanPropertyPointer beanPropertyPointer0 = new BeanPropertyPointer((NodePointer) null, jXPathBasicBeanInfo0);
      boolean boolean0 = beanPropertyPointer0.equals(beanPropertyPointer0);
      assertEquals(Integer.MIN_VALUE, beanPropertyPointer0.getPropertyIndex());
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      NullPropertyPointer nullPropertyPointer0 = new NullPropertyPointer((NodePointer) null);
      Class<NodePointer> class0 = NodePointer.class;
      JXPathBasicBeanInfo jXPathBasicBeanInfo0 = new JXPathBasicBeanInfo(class0);
      BeanPropertyPointer beanPropertyPointer0 = new BeanPropertyPointer(nullPropertyPointer0, jXPathBasicBeanInfo0);
      boolean boolean0 = beanPropertyPointer0.equals(nullPropertyPointer0);
      assertEquals(Integer.MIN_VALUE, beanPropertyPointer0.getPropertyIndex());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      Class<Integer> class0 = Integer.class;
      JXPathBasicBeanInfo jXPathBasicBeanInfo0 = new JXPathBasicBeanInfo(class0);
      BeanPropertyPointer beanPropertyPointer0 = new BeanPropertyPointer((NodePointer) null, jXPathBasicBeanInfo0);
      BeanPropertyPointer beanPropertyPointer1 = new BeanPropertyPointer(beanPropertyPointer0, jXPathBasicBeanInfo0);
      boolean boolean0 = beanPropertyPointer0.equals(beanPropertyPointer1);
      assertEquals(Integer.MIN_VALUE, beanPropertyPointer1.getPropertyIndex());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      NullPropertyPointer nullPropertyPointer0 = new NullPropertyPointer((NodePointer) null);
      Class<NodePointer> class0 = NodePointer.class;
      JXPathBasicBeanInfo jXPathBasicBeanInfo0 = new JXPathBasicBeanInfo(class0);
      BeanPropertyPointer beanPropertyPointer0 = new BeanPropertyPointer(nullPropertyPointer0, jXPathBasicBeanInfo0);
      BeanPropertyPointer beanPropertyPointer1 = new BeanPropertyPointer((NodePointer) null, jXPathBasicBeanInfo0);
      NullPropertyPointer nullPropertyPointer1 = new NullPropertyPointer(beanPropertyPointer1);
      boolean boolean0 = beanPropertyPointer0.equals(nullPropertyPointer1);
      assertFalse(nullPropertyPointer1.equals((Object)nullPropertyPointer0));
      assertTrue(boolean0);
      assertEquals(Integer.MIN_VALUE, nullPropertyPointer1.getPropertyIndex());
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      BeanPropertyPointer beanPropertyPointer0 = new BeanPropertyPointer((NodePointer) null, (JXPathBeanInfo) null);
      beanPropertyPointer0.setPropertyIndex(3);
      NullPropertyPointer nullPropertyPointer0 = new NullPropertyPointer((NodePointer) null);
      boolean boolean0 = beanPropertyPointer0.equals(nullPropertyPointer0);
      assertEquals(3, beanPropertyPointer0.getPropertyIndex());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      NullPropertyPointer nullPropertyPointer0 = new NullPropertyPointer((NodePointer) null);
      nullPropertyPointer0.setPropertyName("<<unknown namespace>>");
      Class<String> class0 = String.class;
      JXPathBasicBeanInfo jXPathBasicBeanInfo0 = new JXPathBasicBeanInfo(class0, class0);
      BeanPropertyPointer beanPropertyPointer0 = new BeanPropertyPointer((NodePointer) null, jXPathBasicBeanInfo0);
      boolean boolean0 = beanPropertyPointer0.equals(nullPropertyPointer0);
      assertFalse(boolean0);
      assertEquals(Integer.MIN_VALUE, beanPropertyPointer0.getPropertyIndex());
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      Class<String> class0 = String.class;
      JXPathBasicBeanInfo jXPathBasicBeanInfo0 = new JXPathBasicBeanInfo(class0);
      BeanPropertyPointer beanPropertyPointer0 = new BeanPropertyPointer((NodePointer) null, jXPathBasicBeanInfo0);
      beanPropertyPointer0.setIndex(177);
      BeanPropertyPointer beanPropertyPointer1 = new BeanPropertyPointer((NodePointer) null, jXPathBasicBeanInfo0);
      boolean boolean0 = beanPropertyPointer0.equals(beanPropertyPointer1);
      assertFalse(beanPropertyPointer1.equals((Object)beanPropertyPointer0));
      assertEquals(Integer.MIN_VALUE, beanPropertyPointer1.getPropertyIndex());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      Class<Integer> class0 = Integer.class;
      JXPathBasicBeanInfo jXPathBasicBeanInfo0 = new JXPathBasicBeanInfo(class0);
      BeanPropertyPointer beanPropertyPointer0 = new BeanPropertyPointer((NodePointer) null, jXPathBasicBeanInfo0);
      beanPropertyPointer0.setIndex(3);
      BeanPropertyPointer beanPropertyPointer1 = (BeanPropertyPointer)beanPropertyPointer0.clone();
      boolean boolean0 = beanPropertyPointer0.equals(beanPropertyPointer1);
      assertEquals(Integer.MIN_VALUE, beanPropertyPointer1.getPropertyIndex());
      assertTrue(boolean0);
  }
}
