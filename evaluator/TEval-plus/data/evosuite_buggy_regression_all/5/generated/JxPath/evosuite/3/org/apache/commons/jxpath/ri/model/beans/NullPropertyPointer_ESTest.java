/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 23:19:29 GMT 2023
 */

package org.apache.commons.jxpath.ri.model.beans;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.util.LinkedList;
import java.util.Locale;
import org.apache.commons.jxpath.BasicVariables;
import org.apache.commons.jxpath.JXPathContext;
import org.apache.commons.jxpath.Variables;
import org.apache.commons.jxpath.ri.QName;
import org.apache.commons.jxpath.ri.model.NodePointer;
import org.apache.commons.jxpath.ri.model.VariablePointer;
import org.apache.commons.jxpath.ri.model.beans.NullPointer;
import org.apache.commons.jxpath.ri.model.beans.NullPropertyPointer;
import org.apache.commons.jxpath.ri.model.beans.PropertyPointer;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class NullPropertyPointer_ESTest extends NullPropertyPointer_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      NullPropertyPointer nullPropertyPointer0 = new NullPropertyPointer((NodePointer) null);
      boolean boolean0 = nullPropertyPointer0.isNode();
      assertEquals("*", nullPropertyPointer0.getPropertyName());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      Locale locale0 = Locale.GERMANY;
      NodePointer nodePointer0 = NodePointer.newNodePointer((QName) null, locale0, locale0);
      NullPropertyPointer nullPropertyPointer0 = new NullPropertyPointer(nodePointer0);
      // Undeclared exception!
      try { 
        nullPropertyPointer0.createChild((JXPathContext) null, (QName) null, Integer.MIN_VALUE, (Object) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.apache.commons.jxpath.ri.model.beans.PropertyPointer", e);
      }
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      NullPropertyPointer nullPropertyPointer0 = new NullPropertyPointer((NodePointer) null);
      boolean boolean0 = nullPropertyPointer0.isLeaf();
      assertEquals("*", nullPropertyPointer0.getPropertyName());
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      NullPropertyPointer nullPropertyPointer0 = new NullPropertyPointer((NodePointer) null);
      nullPropertyPointer0.getNode();
      assertEquals("*", nullPropertyPointer0.getPropertyName());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      Locale locale0 = Locale.PRC;
      NullPointer nullPointer0 = new NullPointer((QName) null, locale0);
      NullPropertyPointer nullPropertyPointer0 = new NullPropertyPointer(nullPointer0);
      nullPropertyPointer0.setNameAttributeValue("\"substring\"");
      // Undeclared exception!
      try { 
        nullPropertyPointer0.setValue((Object) null);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // Cannot set property null()[@name='&quot;substring&quot;'], path does not match a changeable location
         //
         verifyException("org.apache.commons.jxpath.ri.model.beans.NullPropertyPointer", e);
      }
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      NullPropertyPointer nullPropertyPointer0 = new NullPropertyPointer((NodePointer) null);
      nullPropertyPointer0.setPropertyName("pjjTx");
      assertEquals("pjjTx", nullPropertyPointer0.getPropertyName());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      QName qName0 = new QName("9");
      VariablePointer variablePointer0 = new VariablePointer(qName0);
      NullPropertyPointer nullPropertyPointer0 = new NullPropertyPointer(variablePointer0);
      String[] stringArray0 = nullPropertyPointer0.getPropertyNames();
      assertEquals(0, stringArray0.length);
      assertEquals("*", nullPropertyPointer0.getPropertyName());
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      NullPropertyPointer nullPropertyPointer0 = new NullPropertyPointer((NodePointer) null);
      nullPropertyPointer0.getImmediateNode();
      assertEquals("*", nullPropertyPointer0.getPropertyName());
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      NullPropertyPointer nullPropertyPointer0 = new NullPropertyPointer((NodePointer) null);
      nullPropertyPointer0.getBaseValue();
      assertEquals("*", nullPropertyPointer0.getPropertyName());
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      QName qName0 = new QName("");
      VariablePointer variablePointer0 = new VariablePointer(qName0);
      NullPropertyPointer nullPropertyPointer0 = new NullPropertyPointer(variablePointer0);
      JXPathContext jXPathContext0 = JXPathContext.newContext((Object) nullPropertyPointer0);
      // Undeclared exception!
      try { 
        nullPropertyPointer0.createChild(jXPathContext0, qName0, Integer.MIN_VALUE);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // Factory is not set on the JXPathContext - cannot create path: $
         //
         verifyException("org.apache.commons.jxpath.ri.model.VariablePointer", e);
      }
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      BasicVariables basicVariables0 = new BasicVariables();
      VariablePointer variablePointer0 = new VariablePointer(basicVariables0, (QName) null);
      NullPointer nullPointer0 = new NullPointer(variablePointer0, (QName) null);
      NullPropertyPointer nullPropertyPointer0 = new NullPropertyPointer(nullPointer0);
      boolean boolean0 = nullPropertyPointer0.isActual();
      assertFalse(boolean0);
      assertEquals("*", nullPropertyPointer0.getPropertyName());
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      Locale locale0 = Locale.JAPANESE;
      NullPointer nullPointer0 = new NullPointer(locale0, "");
      PropertyPointer propertyPointer0 = nullPointer0.getPropertyPointer();
      propertyPointer0.setPropertyIndex(Integer.MIN_VALUE);
      assertEquals("*", propertyPointer0.getPropertyName());
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      QName qName0 = new QName((String) null, (String) null);
      NodePointer nodePointer0 = NodePointer.newNodePointer(qName0, (Object) null, (Locale) null);
      NullPropertyPointer nullPropertyPointer0 = new NullPropertyPointer(nodePointer0);
      int int0 = nullPropertyPointer0.getLength();
      assertEquals("*", nullPropertyPointer0.getPropertyName());
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      Locale locale0 = Locale.CANADA_FRENCH;
      NullPointer nullPointer0 = new NullPointer(locale0, "y)G7n");
      NullPropertyPointer nullPropertyPointer0 = new NullPropertyPointer(nullPointer0);
      int int0 = nullPropertyPointer0.getPropertyCount();
      assertEquals("*", nullPropertyPointer0.getPropertyName());
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      QName qName0 = new QName("2es[,ofX$Y@Bj?n", "2es[,ofX$Y@Bj?n");
      Locale locale0 = Locale.US;
      NodePointer nodePointer0 = NodePointer.newNodePointer(qName0, "2es[,ofX$Y@Bj?n", locale0);
      NullPropertyPointer nullPropertyPointer0 = new NullPropertyPointer(nodePointer0);
      boolean boolean0 = nullPropertyPointer0.isActualProperty();
      assertEquals("*", nullPropertyPointer0.getPropertyName());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      NullPropertyPointer nullPropertyPointer0 = new NullPropertyPointer((NodePointer) null);
      // Undeclared exception!
      try { 
        nullPropertyPointer0.setValue((Object) null);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // Cannot set property /*, the target object is null
         //
         verifyException("org.apache.commons.jxpath.ri.model.beans.NullPropertyPointer", e);
      }
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      VariablePointer variablePointer0 = new VariablePointer((QName) null);
      NullPropertyPointer nullPropertyPointer0 = new NullPropertyPointer(variablePointer0);
      // Undeclared exception!
      try { 
        nullPropertyPointer0.setValue(variablePointer0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // Cannot set property $null, the target object is null
         //
         verifyException("org.apache.commons.jxpath.ri.model.beans.NullPropertyPointer", e);
      }
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      Locale locale0 = Locale.TAIWAN;
      LinkedList<String> linkedList0 = new LinkedList<String>();
      NodePointer nodePointer0 = NodePointer.newNodePointer((QName) null, linkedList0, locale0);
      NullPropertyPointer nullPropertyPointer0 = new NullPropertyPointer(nodePointer0);
      // Undeclared exception!
      try { 
        nullPropertyPointer0.setValue(linkedList0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // Cannot set property /*, path does not match a changeable location
         //
         verifyException("org.apache.commons.jxpath.ri.model.beans.NullPropertyPointer", e);
      }
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      Locale locale0 = Locale.SIMPLIFIED_CHINESE;
      NodePointer nodePointer0 = NodePointer.newNodePointer((QName) null, locale0, locale0);
      NullPropertyPointer nullPropertyPointer0 = new NullPropertyPointer(nodePointer0);
      nullPropertyPointer0.setAttribute(true);
      // Undeclared exception!
      try { 
        nullPropertyPointer0.createPath((JXPathContext) null);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // Cannot create an attribute for path //@*, operation is not allowed for this type of node
         //
         verifyException("org.apache.commons.jxpath.ri.model.NodePointer", e);
      }
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      BasicVariables basicVariables0 = new BasicVariables();
      QName qName0 = new QName("");
      VariablePointer variablePointer0 = new VariablePointer(basicVariables0, qName0);
      NullPropertyPointer nullPropertyPointer0 = new NullPropertyPointer(variablePointer0);
      // Undeclared exception!
      try { 
        nullPropertyPointer0.createPath((JXPathContext) null);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // No such variable: ''
         //
         verifyException("org.apache.commons.jxpath.BasicVariables", e);
      }
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      Locale locale0 = Locale.KOREA;
      QName qName0 = new QName("+s@Sg", "Cannot set property ");
      JXPathContext jXPathContext0 = JXPathContext.newContext((Object) locale0);
      Variables variables0 = jXPathContext0.getVariables();
      VariablePointer variablePointer0 = new VariablePointer(variables0, qName0);
      NullPropertyPointer nullPropertyPointer0 = new NullPropertyPointer(variablePointer0);
      // Undeclared exception!
      try { 
        nullPropertyPointer0.createPath(jXPathContext0, (Object) nullPropertyPointer0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // No such variable: '+s@Sg:Cannot set property '
         //
         verifyException("org.apache.commons.jxpath.BasicVariables", e);
      }
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      Locale locale0 = Locale.KOREA;
      QName qName0 = new QName("+s@Sg", "Cannot set property ");
      JXPathContext jXPathContext0 = JXPathContext.newContext((Object) locale0);
      Variables variables0 = jXPathContext0.getVariables();
      VariablePointer variablePointer0 = new VariablePointer(variables0, qName0);
      NullPropertyPointer nullPropertyPointer0 = new NullPropertyPointer(variablePointer0);
      nullPropertyPointer0.setAttribute(true);
      // Undeclared exception!
      try { 
        nullPropertyPointer0.createPath(jXPathContext0, (Object) variablePointer0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // Cannot create an attribute for path $+s@Sg:Cannot set property /@*, operation is not allowed for this type of node
         //
         verifyException("org.apache.commons.jxpath.ri.model.NodePointer", e);
      }
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      Locale locale0 = Locale.SIMPLIFIED_CHINESE;
      NodePointer nodePointer0 = NodePointer.newNodePointer((QName) null, locale0, locale0);
      NullPropertyPointer nullPropertyPointer0 = new NullPropertyPointer(nodePointer0);
      // Undeclared exception!
      try { 
        nullPropertyPointer0.createPath((JXPathContext) null, (Object) locale0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // Cannot set property: /* - no such property
         //
         verifyException("org.apache.commons.jxpath.ri.model.beans.BeanPropertyPointer", e);
      }
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      Locale locale0 = Locale.SIMPLIFIED_CHINESE;
      NodePointer nodePointer0 = NodePointer.newNodePointer((QName) null, locale0, locale0);
      NullPropertyPointer nullPropertyPointer0 = new NullPropertyPointer(nodePointer0);
      boolean boolean0 = nullPropertyPointer0.isCollection();
      assertEquals("*", nullPropertyPointer0.getPropertyName());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      Locale locale0 = Locale.CANADA_FRENCH;
      NodePointer nodePointer0 = NodePointer.newNodePointer((QName) null, (Object) null, locale0);
      NullPropertyPointer nullPropertyPointer0 = new NullPropertyPointer(nodePointer0);
      nullPropertyPointer0.setIndex(900);
      // Undeclared exception!
      try { 
        nullPropertyPointer0.setValue(nodePointer0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // Cannot set property null()/*[901], path does not match a changeable location
         //
         verifyException("org.apache.commons.jxpath.ri.model.beans.NullPropertyPointer", e);
      }
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      Locale locale0 = Locale.SIMPLIFIED_CHINESE;
      NullPointer nullPointer0 = new NullPointer(locale0, (String) null);
      NullPropertyPointer nullPropertyPointer0 = new NullPropertyPointer(nullPointer0);
      nullPropertyPointer0.setIndex(107);
      nullPropertyPointer0.setNameAttributeValue("node()");
      nullPropertyPointer0.asPath();
      assertEquals("node()", nullPropertyPointer0.getPropertyName());
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      Locale locale0 = Locale.SIMPLIFIED_CHINESE;
      NodePointer nodePointer0 = NodePointer.newNodePointer((QName) null, locale0, locale0);
      NullPropertyPointer nullPropertyPointer0 = new NullPropertyPointer(nodePointer0);
      nullPropertyPointer0.setNameAttributeValue("K4<#_Ee'");
      nullPropertyPointer0.asPath();
      assertEquals("K4<#_Ee'", nullPropertyPointer0.getPropertyName());
  }
}
