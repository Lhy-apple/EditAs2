/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 21:59:13 GMT 2023
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
import org.jdom.Namespace;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class JDOMAttributeIterator_ESTest extends JDOMAttributeIterator_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      QName qName0 = new QName("-YL}Nd");
      Object object0 = new Object();
      Locale locale0 = new Locale("-YL}Nd", "-YL}Nd", "-YL}Nd");
      NodePointer nodePointer0 = NodePointer.newNodePointer(qName0, object0, locale0);
      JDOMAttributeIterator jDOMAttributeIterator0 = new JDOMAttributeIterator(nodePointer0, qName0);
      int int0 = jDOMAttributeIterator0.getPosition();
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      QName qName0 = new QName("y", "y");
      Element element0 = new Element("y");
      NodePointer nodePointer0 = NodePointer.newNodePointer(qName0, element0, (Locale) null);
      QName qName1 = new QName("xml", "xml");
      JDOMAttributeIterator jDOMAttributeIterator0 = new JDOMAttributeIterator(nodePointer0, qName1);
      assertEquals(0, jDOMAttributeIterator0.getPosition());
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      QName qName0 = new QName("y", "y");
      Element element0 = new Element("y", "y", "y");
      Element element1 = element0.setAttribute("y", "y");
      NodePointer nodePointer0 = NodePointer.newNodePointer(qName0, element1, (Locale) null);
      QName qName1 = new QName("*");
      JDOMAttributeIterator jDOMAttributeIterator0 = new JDOMAttributeIterator(nodePointer0, qName1);
      assertEquals(0, jDOMAttributeIterator0.getPosition());
  }

  @Test(timeout = 4000)
  public void test3()  throws Throwable  {
      QName qName0 = new QName("nP");
      Element element0 = new Element("nP");
      NodePointer nodePointer0 = NodePointer.newNodePointer(qName0, element0, (Locale) null);
      JXPathContext jXPathContext0 = JXPathContext.newContext((Object) element0);
      NodePointer nodePointer1 = nodePointer0.createAttribute(jXPathContext0, qName0);
      assertTrue(nodePointer1.isActual());
  }

  @Test(timeout = 4000)
  public void test4()  throws Throwable  {
      QName qName0 = new QName("nP", "nP");
      Element element0 = new Element("nP", "nP", "nP");
      NodePointer nodePointer0 = NodePointer.newNodePointer(qName0, element0, (Locale) null);
      JDOMAttributeIterator jDOMAttributeIterator0 = new JDOMAttributeIterator(nodePointer0, qName0);
      assertEquals(0, jDOMAttributeIterator0.getPosition());
      
      jDOMAttributeIterator0.getNodePointer();
      assertEquals(1, jDOMAttributeIterator0.getPosition());
  }

  @Test(timeout = 4000)
  public void test5()  throws Throwable  {
      QName qName0 = new QName("y");
      Namespace namespace0 = Namespace.NO_NAMESPACE;
      Element element0 = new Element("y");
      Element element1 = element0.setAttribute("y", "y", namespace0);
      NodePointer nodePointer0 = NodePointer.newNodePointer(qName0, element1, (Locale) null);
      JDOMAttributeIterator jDOMAttributeIterator0 = new JDOMAttributeIterator(nodePointer0, qName0);
      NodePointer nodePointer1 = jDOMAttributeIterator0.getNodePointer();
      assertNotNull(nodePointer1);
      assertEquals(0, jDOMAttributeIterator0.getPosition());
  }

  @Test(timeout = 4000)
  public void test6()  throws Throwable  {
      QName qName0 = new QName("JGpn#");
      NodePointer nodePointer0 = NodePointer.newChildNodePointer((NodePointer) null, qName0, (Object) null);
      JDOMAttributeIterator jDOMAttributeIterator0 = new JDOMAttributeIterator(nodePointer0, qName0);
      NodePointer nodePointer1 = jDOMAttributeIterator0.getNodePointer();
      assertNull(nodePointer1);
  }

  @Test(timeout = 4000)
  public void test7()  throws Throwable  {
      QName qName0 = new QName("y", "y");
      Element element0 = new Element("y");
      NodePointer nodePointer0 = NodePointer.newNodePointer(qName0, element0, (Locale) null);
      JDOMAttributeIterator jDOMAttributeIterator0 = new JDOMAttributeIterator(nodePointer0, qName0);
      boolean boolean0 = jDOMAttributeIterator0.setPosition((-703));
      assertEquals((-703), jDOMAttributeIterator0.getPosition());
      assertFalse(boolean0);
  }
}
