/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 23:20:12 GMT 2023
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
import org.apache.html.dom.HTMLHtmlElementImpl;
import org.apache.html.dom.HTMLIsIndexElementImpl;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class DOMAttributeIterator_ESTest extends DOMAttributeIterator_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
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
  public void test1()  throws Throwable  {
      QName qName0 = new QName("*", "*");
      HTMLDocumentImpl hTMLDocumentImpl0 = new HTMLDocumentImpl();
      Locale locale0 = Locale.TAIWAN;
      NodePointer nodePointer0 = NodePointer.newNodePointer(qName0, hTMLDocumentImpl0, locale0);
      DOMAttributeIterator dOMAttributeIterator0 = new DOMAttributeIterator(nodePointer0, qName0);
      assertEquals(0, dOMAttributeIterator0.getPosition());
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      QName qName0 = new QName("*", "*");
      HTMLDocumentImpl hTMLDocumentImpl0 = new HTMLDocumentImpl();
      HTMLHtmlElementImpl hTMLHtmlElementImpl0 = new HTMLHtmlElementImpl(hTMLDocumentImpl0, "*");
      Locale locale0 = Locale.ENGLISH;
      NodePointer nodePointer0 = NodePointer.newNodePointer(qName0, hTMLHtmlElementImpl0, locale0);
      DOMAttributeIterator dOMAttributeIterator0 = new DOMAttributeIterator(nodePointer0, qName0);
      assertEquals(0, dOMAttributeIterator0.getPosition());
      
      dOMAttributeIterator0.getNodePointer();
      assertEquals(1, dOMAttributeIterator0.getPosition());
  }

  @Test(timeout = 4000)
  public void test3()  throws Throwable  {
      QName qName0 = new QName("xmlns", "xmlns");
      HTMLDocumentImpl hTMLDocumentImpl0 = new HTMLDocumentImpl();
      HTMLHtmlElementImpl hTMLHtmlElementImpl0 = new HTMLHtmlElementImpl(hTMLDocumentImpl0, "xmlns");
      Locale locale0 = new Locale("xmlns", "xmlns", "xmlns");
      NodePointer nodePointer0 = NodePointer.newNodePointer(qName0, "3UW)o,Tf", locale0);
      NodePointer nodePointer1 = NodePointer.newChildNodePointer(nodePointer0, qName0, hTMLHtmlElementImpl0);
      NodePointer nodePointer2 = nodePointer1.createAttribute((JXPathContext) null, qName0);
      assertFalse(nodePointer2.isAttribute());
  }

  @Test(timeout = 4000)
  public void test4()  throws Throwable  {
      QName qName0 = new QName("*", "*");
      HTMLDocumentImpl hTMLDocumentImpl0 = new HTMLDocumentImpl();
      HTMLHtmlElementImpl hTMLHtmlElementImpl0 = new HTMLHtmlElementImpl(hTMLDocumentImpl0, "*");
      Locale locale0 = Locale.ENGLISH;
      NodePointer nodePointer0 = NodePointer.newNodePointer(qName0, hTMLHtmlElementImpl0, locale0);
      hTMLHtmlElementImpl0.setAttribute("xmlns", (String) null);
      DOMAttributeIterator dOMAttributeIterator0 = new DOMAttributeIterator(nodePointer0, qName0);
      assertEquals(0, dOMAttributeIterator0.getPosition());
  }

  @Test(timeout = 4000)
  public void test5()  throws Throwable  {
      QName qName0 = new QName("xmlns", "xmlns");
      HTMLDocumentImpl hTMLDocumentImpl0 = new HTMLDocumentImpl();
      HTMLHtmlElementImpl hTMLHtmlElementImpl0 = new HTMLHtmlElementImpl(hTMLDocumentImpl0, "xmlns");
      Locale locale0 = Locale.ITALIAN;
      NodePointer nodePointer0 = NodePointer.newNodePointer(qName0, hTMLHtmlElementImpl0, locale0);
      hTMLHtmlElementImpl0.setTitle("<<unknown namespace>>");
      DOMAttributeIterator dOMAttributeIterator0 = new DOMAttributeIterator(nodePointer0, qName0);
      assertEquals(0, dOMAttributeIterator0.getPosition());
  }

  @Test(timeout = 4000)
  public void test6()  throws Throwable  {
      QName qName0 = new QName("*");
      HTMLDocumentImpl hTMLDocumentImpl0 = new HTMLDocumentImpl();
      HTMLHtmlElementImpl hTMLHtmlElementImpl0 = new HTMLHtmlElementImpl(hTMLDocumentImpl0, "*");
      Locale locale0 = Locale.ENGLISH;
      hTMLHtmlElementImpl0.setLang("*");
      NodePointer nodePointer0 = NodePointer.newNodePointer(qName0, hTMLHtmlElementImpl0, locale0);
      DOMAttributeIterator dOMAttributeIterator0 = new DOMAttributeIterator(nodePointer0, qName0);
      assertEquals(0, dOMAttributeIterator0.getPosition());
  }

  @Test(timeout = 4000)
  public void test7()  throws Throwable  {
      QName qName0 = new QName("=7*");
      HTMLDocumentImpl hTMLDocumentImpl0 = new HTMLDocumentImpl();
      HTMLHtmlElementImpl hTMLHtmlElementImpl0 = new HTMLHtmlElementImpl(hTMLDocumentImpl0, "=7*");
      Locale locale0 = Locale.ENGLISH;
      NodePointer nodePointer0 = NodePointer.newNodePointer(qName0, hTMLHtmlElementImpl0, locale0);
      DOMAttributeIterator dOMAttributeIterator0 = new DOMAttributeIterator(nodePointer0, qName0);
      assertEquals(0, dOMAttributeIterator0.getPosition());
  }

  @Test(timeout = 4000)
  public void test8()  throws Throwable  {
      QName qName0 = new QName("*", "*");
      HTMLDocumentImpl hTMLDocumentImpl0 = new HTMLDocumentImpl();
      HTMLHtmlElementImpl hTMLHtmlElementImpl0 = new HTMLHtmlElementImpl(hTMLDocumentImpl0, "*");
      Locale locale0 = Locale.ENGLISH;
      hTMLHtmlElementImpl0.setLang("*");
      NodePointer nodePointer0 = NodePointer.newNodePointer(qName0, hTMLHtmlElementImpl0, locale0);
      DOMAttributeIterator dOMAttributeIterator0 = new DOMAttributeIterator(nodePointer0, qName0);
      NodePointer nodePointer1 = dOMAttributeIterator0.getNodePointer();
      assertNotNull(nodePointer1);
      assertEquals(0, dOMAttributeIterator0.getPosition());
  }

  @Test(timeout = 4000)
  public void test9()  throws Throwable  {
      QName qName0 = new QName("", "");
      HTMLDocumentImpl hTMLDocumentImpl0 = new HTMLDocumentImpl();
      HTMLIsIndexElementImpl hTMLIsIndexElementImpl0 = new HTMLIsIndexElementImpl(hTMLDocumentImpl0, "");
      Locale locale0 = Locale.ENGLISH;
      NodePointer nodePointer0 = NodePointer.newNodePointer(qName0, hTMLIsIndexElementImpl0, locale0);
      DOMAttributeIterator dOMAttributeIterator0 = new DOMAttributeIterator(nodePointer0, qName0);
      boolean boolean0 = dOMAttributeIterator0.setPosition(Integer.MIN_VALUE);
      assertEquals(Integer.MIN_VALUE, dOMAttributeIterator0.getPosition());
      assertFalse(boolean0);
  }
}