/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 23:21:04 GMT 2023
 */

package org.apache.commons.jxpath.ri.model.jdom;

import org.junit.Test;
import static org.junit.Assert.*;
import java.util.Locale;
import org.apache.commons.jxpath.JXPathContext;
import org.apache.commons.jxpath.ri.QName;
import org.apache.commons.jxpath.ri.model.NodePointer;
import org.apache.commons.jxpath.ri.model.VariablePointer;
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
      QName qName0 = new QName("false");
      VariablePointer variablePointer0 = new VariablePointer(qName0);
      NodePointer nodePointer0 = NodePointer.newChildNodePointer(variablePointer0, qName0, "false");
      JDOMAttributeIterator jDOMAttributeIterator0 = new JDOMAttributeIterator(nodePointer0, qName0);
      int int0 = jDOMAttributeIterator0.getPosition();
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      QName qName0 = new QName("xml", "xml");
      Locale locale0 = Locale.SIMPLIFIED_CHINESE;
      Element element0 = new Element("xml");
      NodePointer nodePointer0 = NodePointer.newNodePointer(qName0, element0, locale0);
      JDOMAttributeIterator jDOMAttributeIterator0 = new JDOMAttributeIterator(nodePointer0, qName0);
      assertEquals(0, jDOMAttributeIterator0.getPosition());
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      QName qName0 = new QName("_VMqP_", "_VMqP_");
      Locale locale0 = Locale.ITALY;
      Element element0 = new Element("_VMqP_");
      NodePointer nodePointer0 = NodePointer.newNodePointer(qName0, element0, locale0);
      JDOMAttributeIterator jDOMAttributeIterator0 = new JDOMAttributeIterator(nodePointer0, qName0);
      assertEquals(0, jDOMAttributeIterator0.getPosition());
  }

  @Test(timeout = 4000)
  public void test3()  throws Throwable  {
      QName qName0 = new QName("_McqyT", "_McqyT");
      Locale locale0 = Locale.FRENCH;
      Element element0 = new Element("_McqyT", "_McqyT", "_McqyT");
      element0.setAttribute("_McqyT", "_McqyT");
      NodePointer nodePointer0 = NodePointer.newNodePointer(qName0, element0, locale0);
      QName qName1 = new QName("_McqyT", "*");
      JDOMAttributeIterator jDOMAttributeIterator0 = new JDOMAttributeIterator(nodePointer0, qName1);
      assertEquals(0, jDOMAttributeIterator0.getPosition());
  }

  @Test(timeout = 4000)
  public void test4()  throws Throwable  {
      QName qName0 = new QName("_McqyT");
      Locale locale0 = Locale.FRENCH;
      Element element0 = new Element("_McqyT", "_McqyT", "_McqyT");
      element0.setAttribute("_McqyT", "_McqyT");
      NodePointer nodePointer0 = NodePointer.newNodePointer(qName0, element0, locale0);
      QName qName1 = new QName("*");
      JDOMAttributeIterator jDOMAttributeIterator0 = new JDOMAttributeIterator(nodePointer0, qName1);
      assertEquals(0, jDOMAttributeIterator0.getPosition());
  }

  @Test(timeout = 4000)
  public void test5()  throws Throwable  {
      QName qName0 = new QName("_scqPT");
      Locale locale0 = Locale.ITALY;
      Element element0 = new Element("_scqPT", "_scqPT", "_scqPT");
      NodePointer nodePointer0 = NodePointer.newNodePointer(qName0, element0, locale0);
      JXPathContext jXPathContext0 = JXPathContext.newContext((Object) locale0);
      NodePointer nodePointer1 = nodePointer0.createAttribute(jXPathContext0, qName0);
      assertEquals(Integer.MIN_VALUE, NodePointer.WHOLE_COLLECTION);
  }

  @Test(timeout = 4000)
  public void test6()  throws Throwable  {
      QName qName0 = new QName("_McqPT", "_McqPT");
      Locale locale0 = Locale.CHINA;
      Element element0 = new Element("_McqPT", "_McqPT", "_McqPT");
      NodePointer nodePointer0 = NodePointer.newNodePointer(qName0, element0, locale0);
      JDOMAttributeIterator jDOMAttributeIterator0 = new JDOMAttributeIterator(nodePointer0, qName0);
      assertEquals(0, jDOMAttributeIterator0.getPosition());
      
      jDOMAttributeIterator0.getNodePointer();
      assertEquals(1, jDOMAttributeIterator0.getPosition());
  }

  @Test(timeout = 4000)
  public void test7()  throws Throwable  {
      QName qName0 = new QName("_McqPT");
      Locale locale0 = Locale.JAPANESE;
      Namespace namespace0 = Namespace.getNamespace("_McqPT");
      Element element0 = new Element("_McqPT", namespace0);
      Element element1 = element0.setAttribute("_McqPT", "SRkEhvj7Ag,,^cFm");
      NodePointer nodePointer0 = NodePointer.newNodePointer(qName0, element1, locale0);
      JDOMAttributeIterator jDOMAttributeIterator0 = new JDOMAttributeIterator(nodePointer0, qName0);
      NodePointer nodePointer1 = jDOMAttributeIterator0.getNodePointer();
      assertEquals(0, jDOMAttributeIterator0.getPosition());
      assertNotNull(nodePointer1);
  }

  @Test(timeout = 4000)
  public void test8()  throws Throwable  {
      QName qName0 = new QName("@|'QZjsc!AypP");
      Locale locale0 = Locale.JAPANESE;
      NodePointer nodePointer0 = NodePointer.newNodePointer(qName0, qName0, locale0);
      JDOMAttributeIterator jDOMAttributeIterator0 = new JDOMAttributeIterator(nodePointer0, qName0);
      NodePointer nodePointer1 = jDOMAttributeIterator0.getNodePointer();
      assertNull(nodePointer1);
  }

  @Test(timeout = 4000)
  public void test9()  throws Throwable  {
      QName qName0 = new QName("_McqP_T");
      Locale locale0 = Locale.ITALY;
      Element element0 = new Element("_McqP_T", "_McqP_T");
      NodePointer nodePointer0 = NodePointer.newNodePointer(qName0, element0, locale0);
      JDOMAttributeIterator jDOMAttributeIterator0 = new JDOMAttributeIterator(nodePointer0, qName0);
      boolean boolean0 = jDOMAttributeIterator0.setPosition(Integer.MIN_VALUE);
      assertEquals(Integer.MIN_VALUE, jDOMAttributeIterator0.getPosition());
      assertFalse(boolean0);
  }
}