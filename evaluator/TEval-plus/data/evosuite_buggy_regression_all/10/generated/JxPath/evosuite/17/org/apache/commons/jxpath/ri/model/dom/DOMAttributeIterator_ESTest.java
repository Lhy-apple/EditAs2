/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 14:03:45 GMT 2023
 */

package org.apache.commons.jxpath.ri.model.dom;

import org.junit.Test;
import static org.junit.Assert.*;
import java.util.Locale;
import org.apache.commons.jxpath.JXPathContext;
import org.apache.commons.jxpath.ri.QName;
import org.apache.commons.jxpath.ri.model.NodePointer;
import org.apache.commons.jxpath.ri.model.dom.DOMAttributeIterator;
import org.apache.html.dom.HTMLDocumentImpl;
import org.apache.html.dom.HTMLIsIndexElementImpl;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class DOMAttributeIterator_ESTest extends DOMAttributeIterator_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      QName qName0 = new QName("", "");
      HTMLDocumentImpl hTMLDocumentImpl0 = new HTMLDocumentImpl();
      HTMLIsIndexElementImpl hTMLIsIndexElementImpl0 = new HTMLIsIndexElementImpl(hTMLDocumentImpl0, "");
      Locale locale0 = Locale.forLanguageTag("");
      NodePointer nodePointer0 = NodePointer.newNodePointer(qName0, hTMLIsIndexElementImpl0, locale0);
      DOMAttributeIterator dOMAttributeIterator0 = new DOMAttributeIterator(nodePointer0, qName0);
      int int0 = dOMAttributeIterator0.getPosition();
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      QName qName0 = new QName("xmlns", "xmlns");
      HTMLDocumentImpl hTMLDocumentImpl0 = new HTMLDocumentImpl();
      HTMLIsIndexElementImpl hTMLIsIndexElementImpl0 = new HTMLIsIndexElementImpl(hTMLDocumentImpl0, "xmlns");
      Locale locale0 = Locale.JAPAN;
      NodePointer nodePointer0 = NodePointer.newNodePointer(qName0, hTMLIsIndexElementImpl0, locale0);
      NodePointer nodePointer1 = nodePointer0.createAttribute((JXPathContext) null, qName0);
      assertNotNull(nodePointer1);
      
      DOMAttributeIterator dOMAttributeIterator0 = new DOMAttributeIterator(nodePointer1, qName0);
      assertEquals(0, dOMAttributeIterator0.getPosition());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      QName qName0 = new QName("*", "*");
      HTMLDocumentImpl hTMLDocumentImpl0 = new HTMLDocumentImpl();
      HTMLIsIndexElementImpl hTMLIsIndexElementImpl0 = new HTMLIsIndexElementImpl(hTMLDocumentImpl0, "*");
      Locale locale0 = Locale.forLanguageTag("*");
      NodePointer nodePointer0 = NodePointer.newNodePointer(qName0, hTMLIsIndexElementImpl0, locale0);
      QName qName1 = new QName("xmlns", "*");
      hTMLIsIndexElementImpl0.setPrompt("xmlns");
      DOMAttributeIterator dOMAttributeIterator0 = new DOMAttributeIterator(nodePointer0, qName1);
      assertEquals(0, dOMAttributeIterator0.getPosition());
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      QName qName0 = new QName("xmlns", "xmlns");
      HTMLDocumentImpl hTMLDocumentImpl0 = new HTMLDocumentImpl();
      HTMLIsIndexElementImpl hTMLIsIndexElementImpl0 = new HTMLIsIndexElementImpl(hTMLDocumentImpl0, "xmlns");
      hTMLIsIndexElementImpl0.setAttribute("xmlns", "xmlns");
      Locale locale0 = new Locale("xmlns", "xmlns");
      NodePointer nodePointer0 = NodePointer.newNodePointer(qName0, hTMLIsIndexElementImpl0, locale0);
      DOMAttributeIterator dOMAttributeIterator0 = new DOMAttributeIterator(nodePointer0, qName0);
      assertEquals(0, dOMAttributeIterator0.getPosition());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      QName qName0 = new QName("*", "*");
      HTMLDocumentImpl hTMLDocumentImpl0 = new HTMLDocumentImpl();
      HTMLIsIndexElementImpl hTMLIsIndexElementImpl0 = new HTMLIsIndexElementImpl(hTMLDocumentImpl0, "#");
      Locale locale0 = Locale.ENGLISH;
      NodePointer nodePointer0 = NodePointer.newNodePointer(qName0, hTMLIsIndexElementImpl0, locale0);
      hTMLIsIndexElementImpl0.setPrompt("<<unknown namespace>>");
      QName qName1 = new QName("xmlns", "title");
      DOMAttributeIterator dOMAttributeIterator0 = new DOMAttributeIterator(nodePointer0, qName1);
      assertEquals(0, dOMAttributeIterator0.getPosition());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      QName qName0 = new QName("*", "*");
      HTMLDocumentImpl hTMLDocumentImpl0 = new HTMLDocumentImpl();
      HTMLIsIndexElementImpl hTMLIsIndexElementImpl0 = new HTMLIsIndexElementImpl(hTMLDocumentImpl0, "#");
      Locale locale0 = Locale.ENGLISH;
      NodePointer nodePointer0 = NodePointer.newNodePointer(qName0, hTMLIsIndexElementImpl0, locale0);
      hTMLIsIndexElementImpl0.setTitle("*");
      QName qName1 = new QName("xmlns", "title");
      DOMAttributeIterator dOMAttributeIterator0 = new DOMAttributeIterator(nodePointer0, qName1);
      assertEquals(0, dOMAttributeIterator0.getPosition());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      QName qName0 = new QName("*");
      HTMLDocumentImpl hTMLDocumentImpl0 = new HTMLDocumentImpl();
      HTMLIsIndexElementImpl hTMLIsIndexElementImpl0 = new HTMLIsIndexElementImpl(hTMLDocumentImpl0, "*");
      Locale locale0 = Locale.forLanguageTag("*");
      NodePointer nodePointer0 = NodePointer.newNodePointer(qName0, hTMLIsIndexElementImpl0, locale0);
      hTMLIsIndexElementImpl0.setPrompt("http://www.w3.org/2000/xmlns/");
      DOMAttributeIterator dOMAttributeIterator0 = new DOMAttributeIterator(nodePointer0, qName0);
      assertEquals(0, dOMAttributeIterator0.getPosition());
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      QName qName0 = new QName("xmlns");
      HTMLDocumentImpl hTMLDocumentImpl0 = new HTMLDocumentImpl();
      HTMLIsIndexElementImpl hTMLIsIndexElementImpl0 = new HTMLIsIndexElementImpl(hTMLDocumentImpl0, "xmlns");
      Locale locale0 = Locale.CHINESE;
      NodePointer nodePointer0 = NodePointer.newNodePointer(qName0, hTMLIsIndexElementImpl0, locale0);
      DOMAttributeIterator dOMAttributeIterator0 = new DOMAttributeIterator(nodePointer0, qName0);
      assertEquals(0, dOMAttributeIterator0.getPosition());
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      QName qName0 = new QName("*", "*");
      HTMLDocumentImpl hTMLDocumentImpl0 = new HTMLDocumentImpl();
      HTMLIsIndexElementImpl hTMLIsIndexElementImpl0 = new HTMLIsIndexElementImpl(hTMLDocumentImpl0, "*");
      Locale locale0 = Locale.forLanguageTag("*");
      NodePointer nodePointer0 = NodePointer.newNodePointer(qName0, hTMLIsIndexElementImpl0, locale0);
      DOMAttributeIterator dOMAttributeIterator0 = new DOMAttributeIterator(nodePointer0, qName0);
      assertEquals(0, dOMAttributeIterator0.getPosition());
      
      dOMAttributeIterator0.getNodePointer();
      assertEquals(1, dOMAttributeIterator0.getPosition());
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      QName qName0 = new QName("*", "*");
      HTMLDocumentImpl hTMLDocumentImpl0 = new HTMLDocumentImpl();
      HTMLIsIndexElementImpl hTMLIsIndexElementImpl0 = new HTMLIsIndexElementImpl(hTMLDocumentImpl0, "*");
      Locale locale0 = Locale.forLanguageTag("*");
      NodePointer nodePointer0 = NodePointer.newNodePointer(qName0, hTMLIsIndexElementImpl0, locale0);
      hTMLIsIndexElementImpl0.setPrompt("http://www.w3.org/2000/xmlns/");
      DOMAttributeIterator dOMAttributeIterator0 = new DOMAttributeIterator(nodePointer0, qName0);
      dOMAttributeIterator0.getNodePointer();
      assertEquals(0, dOMAttributeIterator0.getPosition());
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      QName qName0 = new QName("", "");
      HTMLDocumentImpl hTMLDocumentImpl0 = new HTMLDocumentImpl();
      HTMLIsIndexElementImpl hTMLIsIndexElementImpl0 = new HTMLIsIndexElementImpl(hTMLDocumentImpl0, "");
      Locale locale0 = Locale.forLanguageTag("");
      NodePointer nodePointer0 = NodePointer.newNodePointer(qName0, hTMLIsIndexElementImpl0, locale0);
      DOMAttributeIterator dOMAttributeIterator0 = new DOMAttributeIterator(nodePointer0, qName0);
      boolean boolean0 = dOMAttributeIterator0.setPosition(Integer.MIN_VALUE);
      assertFalse(boolean0);
  }
}
