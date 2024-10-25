/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 06:19:49 GMT 2023
 */

package org.apache.commons.jxpath.ri.model.jdom;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.util.Locale;
import org.apache.commons.jxpath.ri.QName;
import org.apache.commons.jxpath.ri.model.NodePointer;
import org.apache.commons.jxpath.ri.model.jdom.JDOMAttributeIterator;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.jdom.Element;
import org.jdom.Namespace;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class JDOMAttributeIterator_ESTest extends JDOMAttributeIterator_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      QName qName0 = new QName("iIj.(YkA/!b8", "iIj.(YkA/!b8");
      Locale locale0 = Locale.GERMAN;
      NodePointer nodePointer0 = NodePointer.newNodePointer(qName0, "iIj.(YkA/!b8", locale0);
      JDOMAttributeIterator jDOMAttributeIterator0 = new JDOMAttributeIterator(nodePointer0, qName0);
      int int0 = jDOMAttributeIterator0.getPosition();
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      QName qName0 = new QName("org.apache.commons.jxpath.JXPthContextFactory");
      Locale locale0 = Locale.SIMPLIFIED_CHINESE;
      Element element0 = new Element("org.apache.commons.jxpath.JXPthContextFactory", "org.apache.commons.jxpath.JXPthContextFactory", "org.apache.commons.jxpath.JXPthContextFactory");
      NodePointer nodePointer0 = NodePointer.newNodePointer(qName0, element0, locale0);
      QName qName1 = new QName("xml", (String) null);
      JDOMAttributeIterator jDOMAttributeIterator0 = null;
      try {
        jDOMAttributeIterator0 = new JDOMAttributeIterator(nodePointer0, qName1);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
      }
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      QName qName0 = new QName("org.apache.commons.jxpath.JXPthContextFactory", "org.apache.commons.jxpath.JXPthContextFactory");
      Locale locale0 = Locale.GERMAN;
      Element element0 = new Element("org.apache.commons.jxpath.JXPthContextFactory");
      NodePointer nodePointer0 = NodePointer.newNodePointer(qName0, element0, locale0);
      JDOMAttributeIterator jDOMAttributeIterator0 = new JDOMAttributeIterator(nodePointer0, qName0);
      jDOMAttributeIterator0.getNodePointer();
      // Undeclared exception!
      try { 
        jDOMAttributeIterator0.getNodePointer();
        fail("Expecting exception: IndexOutOfBoundsException");
      
      } catch(IndexOutOfBoundsException e) {
         //
         // Index: 0
         //
         verifyException("java.util.Collections$EmptyList", e);
      }
  }

  @Test(timeout = 4000)
  public void test3()  throws Throwable  {
      QName qName0 = new QName("org.apache.comqons.jPpath.JXPthContextFactory", "org.apache.comqons.jPpath.JXPthContextFactory");
      Locale locale0 = Locale.ENGLISH;
      Element element0 = new Element("org.apache.comqons.jPpath.JXPthContextFactory", "org.apache.comqons.jPpath.JXPthContextFactory", "org.apache.comqons.jPpath.JXPthContextFactory");
      NodePointer nodePointer0 = NodePointer.newNodePointer(qName0, element0, locale0);
      JDOMAttributeIterator jDOMAttributeIterator0 = new JDOMAttributeIterator(nodePointer0, qName0);
      boolean boolean0 = jDOMAttributeIterator0.setPosition(Integer.MIN_VALUE);
      assertEquals(Integer.MIN_VALUE, jDOMAttributeIterator0.getPosition());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test4()  throws Throwable  {
      QName qName0 = new QName("*");
      Namespace namespace0 = Namespace.XML_NAMESPACE;
      Element element0 = new Element("l6LB1", namespace0);
      Locale locale0 = Locale.ROOT;
      Element element1 = element0.setAttribute("l6LB1", "*");
      NodePointer nodePointer0 = NodePointer.newNodePointer(qName0, element1, locale0);
      JDOMAttributeIterator jDOMAttributeIterator0 = new JDOMAttributeIterator(nodePointer0, qName0);
      assertEquals(0, jDOMAttributeIterator0.getPosition());
  }

  @Test(timeout = 4000)
  public void test5()  throws Throwable  {
      QName qName0 = new QName("*");
      Namespace namespace0 = Namespace.XML_NAMESPACE;
      Element element0 = new Element("org.apache.commonsCjxpath.JXPathContextFactory", namespace0);
      Locale locale0 = Locale.CANADA;
      Element element1 = element0.setAttribute("org.apache.commonsCjxpath.JXPathContextFactory", "*", namespace0);
      NodePointer nodePointer0 = NodePointer.newNodePointer(qName0, element1, locale0);
      JDOMAttributeIterator jDOMAttributeIterator0 = new JDOMAttributeIterator(nodePointer0, qName0);
      assertEquals(0, jDOMAttributeIterator0.getPosition());
  }

  @Test(timeout = 4000)
  public void test6()  throws Throwable  {
      QName qName0 = new QName("org.apache.commons.jath.ri.compiler.TreCompiler");
      Locale locale0 = Locale.forLanguageTag("org.apache.commons.jath.ri.compiler.TreCompiler");
      Element element0 = new Element("org.apache.commons.jath.ri.compiler.TreCompiler", "org.apache.commons.jath.ri.compiler.TreCompiler", "org.apache.commons.jath.ri.compiler.TreCompiler");
      element0.setAttribute("org.apache.commons.jath.ri.compiler.TreCompiler", "org.apache.commons.jath.ri.compiler.TreCompiler");
      NodePointer nodePointer0 = NodePointer.newNodePointer(qName0, element0, locale0);
      JDOMAttributeIterator jDOMAttributeIterator0 = new JDOMAttributeIterator(nodePointer0, qName0);
      NodePointer nodePointer1 = jDOMAttributeIterator0.getNodePointer();
      assertEquals(0, jDOMAttributeIterator0.getPosition());
      assertNotNull(nodePointer1);
  }

  @Test(timeout = 4000)
  public void test7()  throws Throwable  {
      Locale locale0 = Locale.ROOT;
      NodePointer nodePointer0 = NodePointer.newNodePointer((QName) null, locale0, locale0);
      JDOMAttributeIterator jDOMAttributeIterator0 = new JDOMAttributeIterator(nodePointer0, (QName) null);
      NodePointer nodePointer1 = jDOMAttributeIterator0.getNodePointer();
      assertNull(nodePointer1);
  }
}
