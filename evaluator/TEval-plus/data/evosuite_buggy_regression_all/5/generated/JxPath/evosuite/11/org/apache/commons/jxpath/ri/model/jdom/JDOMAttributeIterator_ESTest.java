/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 23:20:22 GMT 2023
 */

package org.apache.commons.jxpath.ri.model.jdom;

import org.junit.Test;
import static org.junit.Assert.*;
import java.util.Locale;
import org.apache.commons.jxpath.JXPathContext;
import org.apache.commons.jxpath.ri.QName;
import org.apache.commons.jxpath.ri.model.NodePointer;
import org.apache.commons.jxpath.ri.model.jdom.JDOMAttributeIterator;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.jdom.Element;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class JDOMAttributeIterator_ESTest extends JDOMAttributeIterator_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      QName qName0 = new QName("A8F1{ H4kiy])UT{i", "A8F1{ H4kiy])UT{i");
      NodePointer nodePointer0 = NodePointer.newNodePointer(qName0, "A8F1{ H4kiy])UT{i", (Locale) null);
      JDOMAttributeIterator jDOMAttributeIterator0 = new JDOMAttributeIterator(nodePointer0, qName0);
      int int0 = jDOMAttributeIterator0.getPosition();
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      QName qName0 = new QName("xml", "xml");
      Element element0 = new Element("xml");
      NodePointer nodePointer0 = NodePointer.newNodePointer(qName0, element0, (Locale) null);
      JDOMAttributeIterator jDOMAttributeIterator0 = new JDOMAttributeIterator(nodePointer0, qName0);
      assertEquals(0, jDOMAttributeIterator0.getPosition());
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      QName qName0 = new QName("org.apache.commons.jxpath.ri.model.jdom.JDOMAttributeIterator", "org.apache.commons.jxpath.ri.model.jdom.JDOMAttributeIterator");
      Element element0 = new Element("org.apache.commons.jxpath.ri.model.jdom.JDOMAttributeIterator");
      NodePointer nodePointer0 = NodePointer.newNodePointer(qName0, element0, (Locale) null);
      JDOMAttributeIterator jDOMAttributeIterator0 = new JDOMAttributeIterator(nodePointer0, qName0);
      assertEquals(0, jDOMAttributeIterator0.getPosition());
  }

  @Test(timeout = 4000)
  public void test3()  throws Throwable  {
      QName qName0 = new QName("XBeanInfo", "XBeanInfo");
      Element element0 = new Element("XBeanInfo", "XBeanInfo", "XBeanInfo");
      element0.setAttribute("XBeanInfo", "XBeanInfo");
      NodePointer nodePointer0 = NodePointer.newNodePointer(qName0, element0, (Locale) null);
      QName qName1 = new QName("XBeanInfo", "*");
      JDOMAttributeIterator jDOMAttributeIterator0 = new JDOMAttributeIterator(nodePointer0, qName1);
      assertEquals(0, jDOMAttributeIterator0.getPosition());
  }

  @Test(timeout = 4000)
  public void test4()  throws Throwable  {
      QName qName0 = new QName("XBeanInfo");
      Element element0 = new Element("XBeanInfo", "XBeanInfo", "XBeanInfo");
      element0.setAttribute("XBeanInfo", "XBeanInfo");
      NodePointer nodePointer0 = NodePointer.newNodePointer(qName0, element0, (Locale) null);
      QName qName1 = new QName("*");
      JDOMAttributeIterator jDOMAttributeIterator0 = new JDOMAttributeIterator(nodePointer0, qName1);
      assertEquals(0, jDOMAttributeIterator0.getPosition());
  }

  @Test(timeout = 4000)
  public void test5()  throws Throwable  {
      QName qName0 = new QName("org.apache.commons.jxpath.ri.model.jdom.JDOMAttributIerator");
      Element element0 = new Element("org.apache.commons.jxpath.ri.model.jdom.JDOMAttributIerator", "org.apache.commons.jxpath.ri.model.jdom.JDOMAttributIerator");
      NodePointer nodePointer0 = NodePointer.newNodePointer(qName0, element0, (Locale) null);
      JXPathContext jXPathContext0 = JXPathContext.newContext((Object) null);
      NodePointer nodePointer1 = nodePointer0.createAttribute(jXPathContext0, qName0);
      assertFalse(nodePointer1.isAttribute());
  }

  @Test(timeout = 4000)
  public void test6()  throws Throwable  {
      QName qName0 = new QName("XBeanInfo");
      Element element0 = new Element("XBeanInfo", "XBeanInfo", "XBeanInfo");
      NodePointer nodePointer0 = NodePointer.newNodePointer(qName0, element0, (Locale) null);
      JDOMAttributeIterator jDOMAttributeIterator0 = new JDOMAttributeIterator(nodePointer0, qName0);
      assertEquals(0, jDOMAttributeIterator0.getPosition());
      
      jDOMAttributeIterator0.getNodePointer();
      assertEquals(1, jDOMAttributeIterator0.getPosition());
  }

  @Test(timeout = 4000)
  public void test7()  throws Throwable  {
      QName qName0 = new QName("XBeanInfo");
      Element element0 = new Element("XBeanInfo", "XBeanInfo", "XBeanInfo");
      element0.setAttribute("XBeanInfo", "XBeanInfo");
      NodePointer nodePointer0 = NodePointer.newNodePointer(qName0, element0, (Locale) null);
      JDOMAttributeIterator jDOMAttributeIterator0 = new JDOMAttributeIterator(nodePointer0, qName0);
      NodePointer nodePointer1 = jDOMAttributeIterator0.getNodePointer();
      assertNotNull(nodePointer1);
      assertEquals(0, jDOMAttributeIterator0.getPosition());
  }

  @Test(timeout = 4000)
  public void test8()  throws Throwable  {
      QName qName0 = new QName("org.apache.commons.jxpath.ri.model.jdom.JDOMAttributeIterator");
      NodePointer nodePointer0 = NodePointer.newNodePointer(qName0, qName0, (Locale) null);
      JDOMAttributeIterator jDOMAttributeIterator0 = new JDOMAttributeIterator(nodePointer0, qName0);
      NodePointer nodePointer1 = jDOMAttributeIterator0.getNodePointer();
      assertNull(nodePointer1);
  }

  @Test(timeout = 4000)
  public void test9()  throws Throwable  {
      QName qName0 = new QName("org.apache.commons.jxpath.ri.model.jdom.JDOMAttributeIterator");
      Element element0 = new Element("org.apache.commons.jxpath.ri.model.jdom.JDOMAttributeIterator", "org.apache.commons.jxpath.ri.model.jdom.JDOMAttributeIterator", "org.apache.commons.jxpath.ri.model.jdom.JDOMAttributeIterator");
      NodePointer nodePointer0 = NodePointer.newNodePointer(qName0, element0, (Locale) null);
      JDOMAttributeIterator jDOMAttributeIterator0 = new JDOMAttributeIterator(nodePointer0, qName0);
      boolean boolean0 = jDOMAttributeIterator0.setPosition((-2133));
      assertEquals((-2133), jDOMAttributeIterator0.getPosition());
      assertFalse(boolean0);
  }
}
